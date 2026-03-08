import json
import re

INPUT_FILE = "../data/processed/artifacts.json"
OUTPUT_FILE = "../data/processed/chunks.json"

MIN_LENGTH = 20
MAX_WORDS = 400
OVERLAP = 50

LOW_SIGNAL_PATTERNS = [
    r'^\+1$',
    r'^lgtm$',
    r'^thanks!?$',
    r'^same issue$'
]


def is_low_signal(text):
    text = text.strip().lower()
    if len(text) < MIN_LENGTH:
        return True
    for pattern in LOW_SIGNAL_PATTERNS:
        if re.match(pattern, text):
            return True
    return False


def chunk_text(text):
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + MAX_WORDS
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += MAX_WORDS - OVERLAP

    return chunks


with open(INPUT_FILE, "r") as f:
    artifacts = json.load(f)

chunks = []
chunk_count = 0

for artifact in artifacts:

    text = artifact.get("text", "").strip()

    if is_low_signal(text):
        continue

    text_chunks = chunk_text(text)

    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "chunk_id": f"{artifact['artifact_id']}_{i}",
            "artifact_id": artifact["artifact_id"],
            "chunk_index": i,
            "issue_title": artifact["issue_title"],
            "issue_state": artifact["issue_state"],
            "author": artifact["author"],
            "timestamp": artifact["timestamp"],
            "text": chunk
        })

        chunk_count += 1


with open(OUTPUT_FILE, "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Generated {chunk_count} chunks.")