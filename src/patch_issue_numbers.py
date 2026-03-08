"""
patch_issue_numbers.py

Run from project root:
    python src/patch_issue_numbers.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS_PATH = Path("data/processed/artifacts.json")
CHUNKS_PATH    = Path("data/processed/chunks.json")
ENTITIES_JSONL = Path("data/processed/extracted_entities.jsonl")
CLAIMS_JSONL   = Path("data/processed/extracted_claims.jsonl")


def main() -> None:
    # 1. Build artifact_id → issue_number
    artifacts = json.loads(ARTIFACTS_PATH.read_text())
    issue_map = {a["artifact_id"]: a["issue_number"] for a in artifacts if a.get("issue_number")}
    log.info(f"Issue map: {len(issue_map)} entries")

    # 2. Patch chunks.json — also build chunk_id → issue_number
    chunks = json.loads(CHUNKS_PATH.read_text())
    chunk_map = {}
    for chunk in chunks:
        num = issue_map.get(chunk["artifact_id"])
        if num is not None:
            chunk["issue_number"] = num
        chunk_map[chunk["chunk_id"]] = num
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2))
    log.info(f"Patched chunks.json — sample: {chunks[1].get('issue_number')}")

    # 3. Patch JSONL files
    for path in [ENTITIES_JSONL, CLAIMS_JSONL]:
        if not path.exists():
            continue
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for ev in rec.get("evidence", []):
                if ev.get("issue_number") is None:
                    # try chunk_id directly
                    num = chunk_map.get(ev.get("chunk_id"))
                    # fallback: artifact_id in evidence
                    if num is None:
                        num = issue_map.get(ev.get("artifact_id"))
                    if num is not None:
                        ev["issue_number"] = num
            out.append(json.dumps(rec))
        path.write_text("\n".join(out) + "\n")
        log.info(f"Patched {path.name}")

    log.info("Done. Now re-run: deduplication.py → graph.py → retrieval.py")


if __name__ == "__main__":
    main()