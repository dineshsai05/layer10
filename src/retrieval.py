"""
retrieval.py

Given a question, queries memory.db and returns a grounded context pack:
  - matched entities
  - matched claims with subject/object names
  - formatted citations (author, issue, excerpt)
  - any detected conflicts

Run from project root:
    python src/retrieval.py

Edit the QUESTIONS list at the bottom to try your own queries.
Results are printed to terminal and saved to data/processed/context_packs.json
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH           = Path("data/processed/memory.db")
OUTPUT_PATH       = Path("data/processed/context_packs.json")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

TOP_K_ENTITIES  = 5
TOP_K_CLAIMS    = 10
MAX_PER_ENTITY  = 3     # max claims returned per entity (diversity)

# Claim type pairs that are considered conflicting
CONFLICT_PAIRS = [
    {"fixes_bug", "reports_bug"},
    {"implements", "requests_feature"},
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "so",
    "and", "but", "or", "if", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "about", "who", "what", "when", "where",
    "why", "how", "which", "this", "that", "it", "i", "you", "we",
}

def extract_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", question.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run graph.py first"
        )
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def search_entities(conn: sqlite3.Connection, fts_query: str) -> list[dict]:
    try:
        rows = conn.execute("""
            SELECT e.*
            FROM entities e
            JOIN entities_fts f ON e.rowid = f.rowid
            WHERE entities_fts MATCH ?
            ORDER BY rank
            LIMIT 20
        """, (fts_query,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def search_claims(conn: sqlite3.Connection, fts_query: str) -> list[dict]:
    try:
        rows = conn.execute("""
            SELECT c.*,
                   es.canonical_name AS subject_name,
                   eo.canonical_name AS object_name
            FROM claims c
            JOIN claims_fts f  ON c.rowid = f.rowid
            JOIN entities es   ON c.subject_id = es.entity_id
            JOIN entities eo   ON c.object_id  = eo.entity_id
            WHERE claims_fts MATCH ?
            AND   c.is_current = 1
            ORDER BY rank
            LIMIT 20
        """, (fts_query,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def alias_lookup(conn: sqlite3.Connection, keyword: str) -> dict | None:
    row = conn.execute("""
        SELECT e.*
        FROM entities e
        JOIN entity_aliases a ON e.entity_id = a.entity_id
        WHERE a.alias = ?
        LIMIT 1
    """, (keyword.lower().strip(),)).fetchone()
    return dict(row) if row else None


def claims_for_entity(conn: sqlite3.Connection, entity_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT c.*,
               es.canonical_name AS subject_name,
               eo.canonical_name AS object_name
        FROM claims c
        JOIN entities es ON c.subject_id = es.entity_id
        JOIN entities eo ON c.object_id  = eo.entity_id
        WHERE (c.subject_id = ? OR c.object_id = ?)
        AND   c.is_current = 1
        LIMIT 15
    """, (entity_id, entity_id)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}

def rank_key(item: dict) -> tuple:
    conf  = CONFIDENCE_RANK.get(item.get("confidence", "medium"), 1)
    ev    = len(json.loads(item.get("evidence", "[]")))
    return (conf, -ev)


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def format_citations(claims: list[dict]) -> list[str]:
    citations = []
    for claim in claims:
        evidence_list = json.loads(claim.get("evidence", "[]"))
        for ev in evidence_list[:2]:
            author    = ev.get("author", "unknown")
            issue_num = ev.get("issue_number", "?")
            timestamp = (ev.get("timestamp") or "")[:10]
            excerpt   = ev.get("excerpt", "")[:100]
            predicate = claim.get("predicate_text", "")
            citations.append(
                f'[{author}, issue #{issue_num}, {timestamp}] "{excerpt}" — {predicate}'
            )
    return citations


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def find_conflicts(claims: list[dict]) -> list[dict]:
    pair_claims: dict[tuple, list[dict]] = {}
    for c in claims:
        key = (c["subject_id"], c["object_id"])
        pair_claims.setdefault(key, []).append(c)

    conflicts = []
    for (subj, obj), group in pair_claims.items():
        types = {c["claim_type"] for c in group}
        for conflict_pair in CONFLICT_PAIRS:
            if conflict_pair.issubset(types):
                conflicts.append({
                    "subject_id": subj,
                    "object_id":  obj,
                    "conflicting_types": list(conflict_pair),
                    "claim_ids": [
                        c["claim_id"] for c in group
                        if c["claim_type"] in conflict_pair
                    ],
                })
    return conflicts


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(conn: sqlite3.Connection, question: str) -> dict:
    keywords = extract_keywords(question)
    if not keywords:
        return {"question": question, "entities": [], "claims": [], "citations": [], "conflicts": []}

    fts_query = " OR ".join(keywords)

    # 1. FTS search
    entity_rows = search_entities(conn, fts_query)
    claim_rows  = search_claims(conn, fts_query)

    # 2. Alias lookup for each keyword
    seen_eids = {r["entity_id"] for r in entity_rows}
    for kw in keywords:
        hit = alias_lookup(conn, kw)
        if hit and hit["entity_id"] not in seen_eids:
            entity_rows.append(hit)
            seen_eids.add(hit["entity_id"])

    # 3. Neighbour expansion — pull claims for each matched entity
    seen_cids = {r["claim_id"] for r in claim_rows}
    for eid in list(seen_eids)[:TOP_K_ENTITIES]:
        for c in claims_for_entity(conn, eid):
            if c["claim_id"] not in seen_cids:
                claim_rows.append(c)
                seen_cids.add(c["claim_id"])

    # 4. Rank
    entity_rows = sorted(entity_rows, key=rank_key)
    claim_rows  = sorted(claim_rows,  key=rank_key)

    # 5. Prune — top K with diversity (max MAX_PER_ENTITY claims per entity)
    top_entities = entity_rows[:TOP_K_ENTITIES]

    entity_claim_count: dict[str, int] = {}
    top_claims = []
    for c in claim_rows:
        subj_count = entity_claim_count.get(c["subject_id"], 0)
        obj_count  = entity_claim_count.get(c["object_id"],  0)
        if max(subj_count, obj_count) >= MAX_PER_ENTITY:
            continue
        entity_claim_count[c["subject_id"]] = subj_count + 1
        entity_claim_count[c["object_id"]]  = obj_count  + 1
        top_claims.append(c)
        if len(top_claims) >= TOP_K_CLAIMS:
            break

    # 6. Citations and conflicts
    citations = format_citations(top_claims)
    conflicts = find_conflicts(top_claims)

    return {
        "question":  question,
        "entities":  top_entities,
        "claims":    top_claims,
        "citations": citations,
        "conflicts": conflicts,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def pretty_print(pack: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Q: {pack['question']}")
    print(f"{'='*60}")

    print(f"\nENTITIES ({len(pack['entities'])})")
    for e in pack["entities"]:
        aliases = json.loads(e.get("aliases", "[]"))
        alias_str = f"  aka: {', '.join(aliases[:3])}" if aliases else ""
        print(f"  [{e['entity_type']}] {e['canonical_name']} ({e['confidence']}){alias_str}")

    print(f"\nCLAIMS ({len(pack['claims'])})")
    for c in pack["claims"]:
        print(f"  {c.get('subject_name','?')}  —[{c['claim_type']}]→  {c.get('object_name','?')}")
        print(f"    \"{c['predicate_text']}\"")

    print(f"\nCITATIONS ({len(pack['citations'])})")
    for cit in pack["citations"]:
        print(f"  {cit}")

    if pack["conflicts"]:
        print(f"\nCONFLICTS ({len(pack['conflicts'])})")
        for conf in pack["conflicts"]:
            print(f"  ⚠ {conf}")


# ---------------------------------------------------------------------------
# Questions to run
# ---------------------------------------------------------------------------

QUESTIONS = [
    "Who reported bugs related to streaming?",
    "What components does the OpenAI integration depend on?",
    "Which issues affect the memory module?",
    "What features were requested for agents?",
    "Which bugs were fixed in LangChain?",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn   = connect()
    result = []

    for question in QUESTIONS:
        log.info(f"Retrieving: {question}")
        pack = retrieve(conn, question)
        pretty_print(pack)
        result.append(pack)

    conn.close()

    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    log.info(f"\nContext packs saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()