"""
deduplication.py

Reads:   data/processed/extracted_entities.jsonl
         data/processed/extracted_claims.jsonl
Writes:  data/processed/canonical_entities.json
         data/processed/canonical_claims.json
         data/processed/merge_log.json

Run from project root:
    python src/deduplication.py
"""

from __future__ import annotations

import json
import logging
import re
import difflib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENTITIES_JSONL    = Path("data/processed/extracted_entities.jsonl")
CLAIMS_JSONL      = Path("data/processed/extracted_claims.jsonl")

CANONICAL_ENTITIES_PATH = Path("data/processed/canonical_entities.json")
CANONICAL_CLAIMS_PATH   = Path("data/processed/canonical_claims.json")
MERGE_LOG_PATH          = Path("data/processed/merge_log.json")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# How similar two entity names must be to be merged (0.0 - 1.0)
# 0.85 is fairly strict — catches typos and minor variations
SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", name.strip().lower())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def names_are_similar(names_a: set[str], names_b: set[str]) -> bool:
    """True if any name in A matches any name in B above the threshold."""
    for a in names_a:
        for b in names_b:
            if a == b:
                return True
            if similarity(a, b) >= SIMILARITY_THRESHOLD:
                return True
    return False


def merge_evidence(evidence_lists: list[list[dict]]) -> list[dict]:
    """Merge multiple evidence lists, deduplicating by chunk_id + excerpt."""
    seen = set()
    merged = []
    for ev_list in evidence_lists:
        for ev in ev_list:
            key = (ev.get("chunk_id", ""), ev.get("excerpt", "")[:50])
            if key not in seen:
                seen.add(key)
                merged.append(ev)
    return merged


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        log.warning(f"{path} not found — returning empty list")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning(f"Skipping malformed line in {path}")
    log.info(f"Loaded {len(records)} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Entity deduplication
# ---------------------------------------------------------------------------

def deduplicate_entities(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Two-pass deduplication:
      Pass 1 — exact match on (entity_type, normalized canonical_name)
      Pass 2 — fuzzy match within each entity_type bucket

    Returns (canonical_entities, merge_log_entries)
    """
    merge_log = []

    # ── Pass 1: exact merge ──────────────────────────────────────────────────
    exact_groups: dict[tuple, list[dict]] = defaultdict(list)
    for ent in entities:
        key = (ent["entity_type"], normalize(ent["canonical_name"]))
        exact_groups[key].append(ent)

    after_pass1: list[dict] = []
    for (etype, ename), group in exact_groups.items():
        if len(group) == 1:
            after_pass1.append(group[0])
            continue

        # merge into the first record
        canonical = dict(group[0])
        all_aliases  = set(canonical.get("aliases", []))
        all_evidence = [canonical.get("evidence", [])]
        merged_ids   = [canonical["entity_id"]]

        for ent in group[1:]:
            all_aliases.update(ent.get("aliases", []))
            all_evidence.append(ent.get("evidence", []))
            merged_ids.append(ent["entity_id"])

        canonical["aliases"]  = sorted(all_aliases - {normalize(canonical["canonical_name"])})
        canonical["evidence"] = merge_evidence(all_evidence)
        after_pass1.append(canonical)

        merge_log.append({
            "merge_type":     "exact",
            "canonical_id":   canonical["entity_id"],
            "canonical_name": canonical["canonical_name"],
            "merged_ids":     merged_ids,
            "merged_at":      datetime.now(timezone.utc).isoformat(),
        })

    # ── Pass 2: fuzzy merge within each entity_type ──────────────────────────
    by_type: dict[str, list[dict]] = defaultdict(list)
    for ent in after_pass1:
        by_type[ent["entity_type"]].append(ent)

    final: list[dict] = []

    for etype, group in by_type.items():
        n = len(group)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        # build name sets
        name_sets = []
        for ent in group:
            names = {normalize(ent["canonical_name"])}
            names.update(normalize(a) for a in ent.get("aliases", []))
            name_sets.append(names)

        for i in range(n):
            for j in range(i + 1, n):
                if names_are_similar(name_sets[i], name_sets[j]):
                    union(i, j)

        # collect fuzzy groups
        fuzzy_groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            fuzzy_groups[find(i)].append(i)

        for root, members in fuzzy_groups.items():
            if len(members) == 1:
                final.append(group[members[0]])
                continue

            # pick the record with most evidence as canonical
            members_sorted = sorted(
                members,
                key=lambda i: len(group[i].get("evidence", [])),
                reverse=True,
            )
            canonical  = dict(group[members_sorted[0]])
            all_aliases  = set(canonical.get("aliases", []))
            all_evidence = [canonical.get("evidence", [])]
            merged_ids   = [canonical["entity_id"]]

            for idx in members_sorted[1:]:
                ent = group[idx]
                all_aliases.add(normalize(ent["canonical_name"]))
                all_aliases.update(normalize(a) for a in ent.get("aliases", []))
                all_evidence.append(ent.get("evidence", []))
                merged_ids.append(ent["entity_id"])

            all_aliases.discard(normalize(canonical["canonical_name"]))
            canonical["aliases"]  = sorted(all_aliases)
            canonical["evidence"] = merge_evidence(all_evidence)
            final.append(canonical)

            merge_log.append({
                "merge_type":         "fuzzy",
                "canonical_id":       canonical["entity_id"],
                "canonical_name":     canonical["canonical_name"],
                "merged_ids":         merged_ids,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "merged_at":          datetime.now(timezone.utc).isoformat(),
            })

    return final, merge_log


# ---------------------------------------------------------------------------
# Claim deduplication
# ---------------------------------------------------------------------------

def build_id_remap(merge_log: list[dict]) -> dict[str, str]:
    """old_entity_id → canonical_entity_id"""
    remap = {}
    for entry in merge_log:
        for old_id in entry.get("merged_ids", []):
            if old_id != entry["canonical_id"]:
                remap[old_id] = entry["canonical_id"]
    return remap


def deduplicate_claims(
    claims: list[dict],
    id_remap: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """
    Merge claims with the same (claim_type, subject_id, object_id)
    after remapping entity IDs to their canonical versions.

    Returns (canonical_claims, merge_log_entries)
    """
    merge_log = []

    def remap(eid: str) -> str:
        return id_remap.get(eid, eid)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for claim in claims:
        key = (
            claim["claim_type"],
            remap(claim["subject_id"]),
            remap(claim["object_id"]),
        )
        groups[key].append(claim)

    canonical_claims = []
    for (ctype, subj, obj), group in groups.items():
        if len(group) == 1:
            c = dict(group[0])
            c["subject_id"] = subj
            c["object_id"]  = obj
            canonical_claims.append(c)
            continue

        # merge: keep longest predicate_text, widen validity window
        merged = dict(group[0])
        merged["subject_id"] = subj
        merged["object_id"]  = obj
        merged["evidence"]   = merge_evidence([g.get("evidence", []) for g in group])
        merged["predicate_text"] = max(
            (g.get("predicate_text", "") for g in group), key=len
        )

        valid_froms  = [g["valid_from"]  for g in group if g.get("valid_from")]
        valid_untils = [g["valid_until"] for g in group if g.get("valid_until")]
        merged["valid_from"]  = min(valid_froms)  if valid_froms  else None
        merged["valid_until"] = max(valid_untils) if valid_untils else None
        merged["is_current"]  = merged["valid_until"] is None

        canonical_claims.append(merged)

        merge_log.append({
            "merge_type":        "claim_dedup",
            "canonical_claim_id": merged["claim_id"],
            "claim_type":        ctype,
            "subject_id":        subj,
            "object_id":         obj,
            "merged_count":      len(group),
            "merged_at":         datetime.now(timezone.utc).isoformat(),
        })

    return canonical_claims, merge_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # load
    raw_entities = load_jsonl(ENTITIES_JSONL)
    raw_claims   = load_jsonl(CLAIMS_JSONL)

    if not raw_entities:
        log.error("No entities found — run extractor.py first")
        return

    # deduplicate entities
    log.info("Deduplicating entities...")
    canonical_entities, entity_merge_log = deduplicate_entities(raw_entities)
    log.info(f"Entities: {len(raw_entities)} raw → {len(canonical_entities)} canonical")

    # build ID remap from entity merges
    id_remap = build_id_remap(entity_merge_log)

    # deduplicate claims
    log.info("Deduplicating claims...")
    canonical_claims, claim_merge_log = deduplicate_claims(raw_claims, id_remap)
    log.info(f"Claims:   {len(raw_claims)} raw → {len(canonical_claims)} canonical")

    # write outputs
    CANONICAL_ENTITIES_PATH.write_text(json.dumps(canonical_entities, indent=2))
    log.info(f"Written → {CANONICAL_ENTITIES_PATH}")

    CANONICAL_CLAIMS_PATH.write_text(json.dumps(canonical_claims, indent=2))
    log.info(f"Written → {CANONICAL_CLAIMS_PATH}")

    MERGE_LOG_PATH.write_text(json.dumps({
        "entity_merges": entity_merge_log,
        "claim_merges":  claim_merge_log,
    }, indent=2))
    log.info(f"Written → {MERGE_LOG_PATH}")

    log.info("─" * 50)
    log.info(f"Entity merges : {len(entity_merge_log)}")
    log.info(f"Claim merges  : {len(claim_merge_log)}")
    log.info("Done.")


if __name__ == "__main__":
    main()