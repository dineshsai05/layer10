# Layer10 Memory Graph

Builds a structured knowledge graph from GitHub Issues and Comments.

## Pipeline

```
data_collection.py        →  data/raw/issues.json, comments.json
parse_artifacts.py        →  data/processed/artifacts.json
prepare_chunks.py         →  data/processed/chunks.json
claim_extraction.py       →  data/processed/extractions.json
extractor.py              →  data/processed/extracted_entities.jsonl
                             data/processed/extracted_claims.jsonl
patch_issue_numbers.py    →  patches issue numbers into chunks + extractions
deduplication.py          →  data/processed/canonical_entities.json
                             data/processed/canonical_claims.json
                             data/processed/merge_log.json
graph.py                  →  data/processed/memory.db
retrieval.py              →  data/processed/context_packs.json
visualization.py          →  data/processed/viz/index.html
```

## Setup

```bash
cp .env.example .env
# fill in GITHUB_TOKEN and GROQ_API_KEY in .env
pip install -r requirements.txt
```

## Run

Run each script from the project root in pipeline order:

```bash
python src/data_collection.py
python src/parse_artifacts.py
python src/prepare_chunks.py
python src/claim_extraction.py
python src/extractor.py
python src/patch_issue_numbers.py
python src/deduplication.py
python src/graph.py
python src/retrieval.py
python src/visualization.py
```

## Utils

```bash
python utils/pretty_json.py <input.json> [output.json]
```
