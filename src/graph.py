"""
graph.py

Reads:   data/processed/canonical_entities.json
         data/processed/canonical_claims.json
         data/processed/merge_log.json
Writes:  data/processed/memory.db  (SQLite)

Run from project root:
    python src/graph.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CANONICAL_ENTITIES_PATH = Path("data/processed/canonical_entities.json")
CANONICAL_CLAIMS_PATH   = Path("data/processed/canonical_claims.json")
MERGE_LOG_PATH          = Path("data/processed/merge_log.json")
DB_PATH                 = Path("data/processed/memory.db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entities (
    entity_id          TEXT PRIMARY KEY,
    entity_type        TEXT NOT NULL,
    canonical_name     TEXT NOT NULL,
    description        TEXT,
    confidence         TEXT NOT NULL DEFAULT 'medium',
    aliases            TEXT NOT NULL DEFAULT '[]',
    evidence           TEXT NOT NULL DEFAULT '[]',
    extraction_version TEXT NOT NULL,
    extracted_at       TEXT NOT NULL,
    updated_at         TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id           TEXT PRIMARY KEY,
    claim_type         TEXT NOT NULL,
    subject_id         TEXT NOT NULL REFERENCES entities(entity_id),
    object_id          TEXT NOT NULL REFERENCES entities(entity_id),
    predicate_text     TEXT NOT NULL,
    valid_from         TEXT,
    valid_until        TEXT,
    is_current         INTEGER NOT NULL DEFAULT 1,
    confidence         TEXT NOT NULL DEFAULT 'medium',
    evidence           TEXT NOT NULL DEFAULT '[]',
    extraction_version TEXT NOT NULL,
    extracted_at       TEXT NOT NULL,
    updated_at         TEXT
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias      TEXT NOT NULL,
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id),
    PRIMARY KEY (alias, entity_id)
);

CREATE TABLE IF NOT EXISTS merge_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    merge_type   TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    merged_ids   TEXT NOT NULL,
    merged_at    TEXT NOT NULL,
    note         TEXT
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_id);
CREATE INDEX IF NOT EXISTS idx_claims_object  ON claims(object_id);
CREATE INDEX IF NOT EXISTS idx_claims_type    ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_current ON claims(is_current);
CREATE INDEX IF NOT EXISTS idx_entities_type  ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_alias_entity   ON entity_aliases(entity_id);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    entity_id    UNINDEXED,
    canonical_name,
    description,
    aliases,
    content='entities',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id       UNINDEXED,
    predicate_text,
    content='claims',
    content_rowid='rowid'
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Upserts
# ---------------------------------------------------------------------------

def upsert_entity(conn: sqlite3.Connection, ent: dict) -> None:
    conn.execute("""
        INSERT INTO entities
            (entity_id, entity_type, canonical_name, description,
             confidence, aliases, evidence, extraction_version, extracted_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(entity_id) DO UPDATE SET
            aliases            = excluded.aliases,
            evidence           = excluded.evidence,
            description        = COALESCE(excluded.description, description),
            updated_at         = excluded.extracted_at
    """, (
        ent["entity_id"],
        ent["entity_type"],
        ent["canonical_name"],
        ent.get("description"),
        ent.get("confidence", "medium"),
        json.dumps(ent.get("aliases", [])),
        json.dumps(ent.get("evidence", [])),
        ent.get("extraction_version", "unknown"),
        ent.get("extracted_at", ""),
    ))

    # populate alias lookup table
    aliases = list(ent.get("aliases", []))
    aliases.append(ent["canonical_name"].lower().strip())
    for alias in set(aliases):
        conn.execute("""
            INSERT OR IGNORE INTO entity_aliases (alias, entity_id)
            VALUES (?, ?)
        """, (alias.lower().strip(), ent["entity_id"]))


def upsert_claim(conn: sqlite3.Connection, clm: dict) -> None:
    # only insert if both entities exist
    subj = conn.execute(
        "SELECT 1 FROM entities WHERE entity_id = ?", (clm["subject_id"],)
    ).fetchone()
    obj = conn.execute(
        "SELECT 1 FROM entities WHERE entity_id = ?", (clm["object_id"],)
    ).fetchone()

    if not subj or not obj:
        log.debug(f"Skipping claim {clm['claim_id']} — dangling entity reference")
        return

    conn.execute("""
        INSERT INTO claims
            (claim_id, claim_type, subject_id, object_id, predicate_text,
             valid_from, valid_until, is_current, confidence,
             evidence, extraction_version, extracted_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(claim_id) DO UPDATE SET
            evidence       = excluded.evidence,
            predicate_text = CASE
                WHEN length(excluded.predicate_text) > length(predicate_text)
                THEN excluded.predicate_text
                ELSE predicate_text
            END,
            is_current     = excluded.is_current,
            updated_at     = excluded.extracted_at
    """, (
        clm["claim_id"],
        clm["claim_type"],
        clm["subject_id"],
        clm["object_id"],
        clm.get("predicate_text", ""),
        clm.get("valid_from"),
        clm.get("valid_until"),
        1 if clm.get("is_current", True) else 0,
        clm.get("confidence", "medium"),
        json.dumps(clm.get("evidence", [])),
        clm.get("extraction_version", "unknown"),
        clm.get("extracted_at", ""),
    ))


def insert_merge_log(conn: sqlite3.Connection, merge_log: dict) -> None:
    for entry in merge_log.get("entity_merges", []):
        conn.execute("""
            INSERT INTO merge_log (merge_type, canonical_id, merged_ids, merged_at, note)
            VALUES (?,?,?,?,?)
        """, (
            entry["merge_type"],
            entry["canonical_id"],
            json.dumps(entry.get("merged_ids", [])),
            entry["merged_at"],
            entry.get("note"),
        ))
    for entry in merge_log.get("claim_merges", []):
        conn.execute("""
            INSERT INTO merge_log (merge_type, canonical_id, merged_ids, merged_at, note)
            VALUES (?,?,?,?,?)
        """, (
            entry["merge_type"],
            entry.get("canonical_claim_id", ""),
            json.dumps([]),
            entry["merged_at"],
            f"claim_dedup: {entry.get('claim_type')} × {entry.get('merged_count')}",
        ))


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO entities_fts(entities_fts) VALUES('rebuild')")
    conn.execute("INSERT INTO claims_fts(claims_fts) VALUES('rebuild')")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(conn: sqlite3.Connection) -> None:
    entity_count  = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    claim_count   = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    current_claims = conn.execute("SELECT COUNT(*) FROM claims WHERE is_current=1").fetchone()[0]

    entity_types = conn.execute(
        "SELECT entity_type, COUNT(*) as n FROM entities GROUP BY entity_type ORDER BY n DESC"
    ).fetchall()
    claim_types = conn.execute(
        "SELECT claim_type, COUNT(*) as n FROM claims GROUP BY claim_type ORDER BY n DESC"
    ).fetchall()

    log.info("─" * 50)
    log.info(f"Entities      : {entity_count}")
    log.info(f"Claims        : {claim_count}  (current: {current_claims})")
    log.info("Entity breakdown:")
    for row in entity_types:
        log.info(f"  {row['entity_type']:<20} {row['n']}")
    log.info("Claim breakdown:")
    for row in claim_types:
        log.info(f"  {row['claim_type']:<25} {row['n']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info(f"Building memory graph → {DB_PATH}")

    entities  = json.loads(CANONICAL_ENTITIES_PATH.read_text())
    claims    = json.loads(CANONICAL_CLAIMS_PATH.read_text())
    merge_log = json.loads(MERGE_LOG_PATH.read_text())

    conn = connect(DB_PATH)

    log.info(f"Inserting {len(entities)} entities...")
    for ent in entities:
        upsert_entity(conn, ent)
    conn.commit()

    log.info(f"Inserting {len(claims)} claims...")
    skipped = 0
    for clm in claims:
        upsert_claim(conn, clm)
    conn.commit()

    log.info("Inserting merge log...")
    insert_merge_log(conn, merge_log)
    conn.commit()

    log.info("Rebuilding FTS indices...")
    rebuild_fts(conn)
    conn.commit()

    print_stats(conn)
    conn.close()

    log.info(f"Done.  DB → {DB_PATH}")


if __name__ == "__main__":
    main()