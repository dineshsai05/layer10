"""
extractor.py

Reads:   data/processed/chunks.json
Writes:  data/processed/extracted_entities.jsonl
         data/processed/extracted_claims.jsonl
         data/processed/quarantine.jsonl
         data/processed/checkpoint.txt
         data/processed/extraction_stats.json

Run from project root:
    python src/extractor.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel, Field, field_validator, ValidationError
from enum import Enum
import hashlib

# ---------------------------------------------------------------------------
# Paths  (relative to project root)
# ---------------------------------------------------------------------------

CHUNKS_PATH     = Path("data/processed/chunks.json")
OUTPUT_DIR      = Path("data/processed")

ENTITIES_PATH   = OUTPUT_DIR / "extracted_entities.jsonl"
CLAIMS_PATH     = OUTPUT_DIR / "extracted_claims.jsonl"
QUARANTINE_PATH = OUTPUT_DIR / "quarantine.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.txt"
STATS_PATH      = OUTPUT_DIR / "extraction_stats.json"

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

MAX_CHUNKS         = None       # None = process all chunks
START_OFFSET       = 0          # skip first N chunks (useful for resuming from a batch)

SAVE_INTERVAL_SECS = 600        # flush stats to disk every 10 minutes

MODEL              = "llama-3.1-8b-instant"   # high TPM limit on Groq free tier
MAX_TOKENS         = 2048
EXTRACTION_VERSION = "v1.0.0"   # bump this when you change the prompt or schema

MAX_RETRIES        = 3
RETRY_SLEEP        = 2          # seconds, doubles on each retry
REQUEST_PAUSE      = 0.2        # polite pause between API calls

MIN_EXCERPT_LEN    = 10
CONFIDENCE_GATE    = {"high", "medium"}   # "low" confidence → quarantine

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

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

class EntityType(str, Enum):
    PERSON       = "person"
    COMPONENT    = "component"
    BUG          = "bug"
    FEATURE      = "feature"
    VERSION      = "version"
    DEPENDENCY   = "dependency"
    CONCEPT      = "concept"
    ORGANIZATION = "organization"


class ClaimType(str, Enum):
    REPORTS_BUG       = "reports_bug"
    FIXES_BUG         = "fixes_bug"
    IMPLEMENTS        = "implements"
    REQUESTS_FEATURE  = "requests_feature"
    USES              = "uses"
    DEPENDS_ON        = "depends_on"
    AFFECTS           = "affects"
    INTRODUCED_IN     = "introduced_in"
    FIXED_IN          = "fixed_in"
    MEMBER_OF         = "member_of"
    STATE_CHANGE      = "state_change"
    ASSIGNED_TO       = "assigned_to"
    DUPLICATE_OF      = "duplicate_of"
    HAS_PROPERTY      = "has_property"
    DECISION_MADE     = "decision_made"
    WORKAROUND_EXISTS = "workaround_exists"


class Confidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class Evidence(BaseModel):
    chunk_id:     str
    artifact_id:  str
    excerpt:      str
    char_start:   Optional[int] = None
    char_end:     Optional[int] = None
    author:       str
    timestamp:    str
    issue_number: Optional[int] = None

    @field_validator("excerpt")
    @classmethod
    def truncate(cls, v: str) -> str:
        return v[:300]


class ExtractedEntity(BaseModel):
    entity_id:          str
    entity_type:        EntityType
    canonical_name:     str
    aliases:            list[str]       = Field(default_factory=list)
    description:        Optional[str]   = None
    confidence:         Confidence      = Confidence.MEDIUM
    evidence:           list[Evidence]  = Field(..., min_length=1)
    extraction_version: str
    extracted_at:       str


class ExtractedClaim(BaseModel):
    claim_id:           str
    claim_type:         ClaimType
    subject_id:         str
    object_id:          str
    predicate_text:     str
    valid_from:         Optional[str]   = None
    valid_until:        Optional[str]   = None
    is_current:         bool            = True
    confidence:         Confidence      = Confidence.MEDIUM
    evidence:           list[Evidence]  = Field(..., min_length=1)
    extraction_version: str
    extracted_at:       str


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def make_entity_id(entity_type: str, canonical_name: str) -> str:
    key = f"{entity_type}::{canonical_name.lower().strip()}"
    return "ent_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def make_claim_id(claim_type: str, subject_id: str, object_id: str) -> str:
    key = f"{claim_type}::{subject_id}::{object_id}"
    return "clm_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a structured knowledge extraction engine for a long-term memory system.
Read the GitHub issue chunk below and extract typed entities and claims.

Return ONLY a valid JSON object — no markdown fences, no explanation:

{
  "entities": [
    {
      "canonical_name": "normalized name",
      "entity_type": "person | component | bug | feature | version | dependency | concept | organization",
      "aliases": ["other names seen in this chunk"],
      "description": "one sentence or null",
      "confidence": "high | medium | low",
      "excerpt": "verbatim quote from chunk text (≤200 chars) that establishes this entity"
    }
  ],
  "claims": [
    {
      "claim_type": "reports_bug | fixes_bug | implements | requests_feature | uses | depends_on | affects | introduced_in | fixed_in | member_of | state_change | assigned_to | duplicate_of | has_property | decision_made | workaround_exists",
      "subject": "canonical_name of subject entity",
      "object": "canonical_name of object entity",
      "predicate_text": "plain English statement of the claim",
      "confidence": "high | medium | low",
      "excerpt": "verbatim quote from chunk text (≤200 chars) that supports this claim",
      "valid_from": "ISO date or null",
      "valid_until": "ISO date or null"
    }
  ]
}

Rules:
- Only extract entities explicitly named in the chunk.
- Every excerpt must be a verbatim substring of the chunk text.
- Do not invent relationships. If unsure, use confidence=low or omit.
- Max 8 entities, max 10 claims.
- subject and object in claims must both be entities you extracted.
"""

REPAIR_PROMPT = """\
Fix the malformed JSON below so it matches this structure exactly:
{"entities": [...], "claims": [...]}
Return only the corrected JSON, nothing else.
"""


def build_user_message(chunk: dict) -> str:
    return (
        f"chunk_id: {chunk['chunk_id']}\n"
        f"artifact_id: {chunk['artifact_id']}\n"
        f"issue_title: {chunk.get('issue_title', '')}\n"
        f"issue_state: {chunk.get('issue_state', '')}\n"
        f"author: {chunk.get('author', '')}\n"
        f"timestamp: {chunk.get('timestamp', '')}\n\n"
        f"{chunk['text']}"
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set.\n"
            "Run:  export GROQ_API_KEY='gsk_...'"
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

MAX_RATE_LIMIT_WAIT = 60.0   # never wait more than 60s — if still rate limited, chunk is skipped and retried next run


def _retry_after(exc: RateLimitError) -> Optional[float]:
    """
    Read retry-after seconds from Groq 429 response.
    Caps at MAX_RATE_LIMIT_WAIT so we never freeze for 19 minutes.
    """
    try:
        headers = exc.response.headers
        value   = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        if value:
            return min(float(value) + 1.0, MAX_RATE_LIMIT_WAIT)
    except Exception:
        pass
    match = re.search(r"try again in ([0-9.]+)s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, MAX_RATE_LIMIT_WAIT)
    return None


def call_llm(client: OpenAI, system: str, user: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            wait = _retry_after(e) or min(RETRY_SLEEP * (2 ** attempt), MAX_RATE_LIMIT_WAIT)
            log.warning(f"429 rate limit — waiting {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

        except APIError as e:
            log.error(f"API error on attempt {attempt}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP * attempt)

    # chunk will land in quarantine and checkpoint — safe to re-run later
    raise RuntimeError("Rate limited on all retries — chunk skipped")


def parse_json(raw: str) -> Optional[dict]:
    # strip fences
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # fallback: grab first {...} block in case of leading prose
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def find_offsets(text: str, excerpt: str) -> tuple[Optional[int], Optional[int]]:
    idx = text.find(excerpt)
    if idx == -1:
        idx = text.lower().find(excerpt.lower())
    if idx == -1:
        return None, None
    return idx, idx + len(excerpt)


def make_evidence(chunk: dict, excerpt: str) -> Evidence:
    start, end = find_offsets(chunk.get("text", ""), excerpt)
    return Evidence(
        chunk_id=chunk["chunk_id"],
        artifact_id=chunk["artifact_id"],
        excerpt=excerpt[:300],
        char_start=start,
        char_end=end,
        author=chunk.get("author", "unknown"),
        timestamp=chunk.get("timestamp", ""),
        issue_number=chunk.get("issue_number"),
    )


def excerpt_is_grounded(excerpt: str, chunk_text: str) -> bool:
    if not excerpt or len(excerpt) < MIN_EXCERPT_LEN:
        return False
    return excerpt.lower() in chunk_text.lower()


# ---------------------------------------------------------------------------
# Entity + claim builders
# ---------------------------------------------------------------------------

def build_entity(raw: dict, chunk: dict, extracted_at: str) -> Optional[ExtractedEntity]:
    try:
        entity_type = EntityType(raw.get("entity_type", "concept").lower())
    except ValueError:
        return None

    canonical_name = normalize_name(raw.get("canonical_name", ""))
    if not canonical_name:
        return None

    confidence_str = raw.get("confidence", "medium").lower()
    if confidence_str not in CONFIDENCE_GATE:
        return None  # quarantine low-confidence entities

    try:
        confidence = Confidence(confidence_str)
    except ValueError:
        confidence = Confidence.MEDIUM

    excerpt = raw.get("excerpt", "")[:300]
    if not excerpt_is_grounded(excerpt, chunk.get("text", "")):
        log.debug(f"Excerpt not grounded for '{canonical_name}' — using name as fallback")
        excerpt = canonical_name

    try:
        return ExtractedEntity(
            entity_id=make_entity_id(entity_type.value, canonical_name),
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=[a.lower().strip() for a in raw.get("aliases", []) if a],
            description=raw.get("description"),
            confidence=confidence,
            evidence=[make_evidence(chunk, excerpt)],
            extraction_version=EXTRACTION_VERSION,
            extracted_at=extracted_at,
        )
    except ValidationError:
        return None


def build_claim(
    raw: dict,
    chunk: dict,
    name_to_id: dict[str, str],
    extracted_at: str,
) -> Optional[ExtractedClaim]:
    try:
        claim_type = ClaimType(raw.get("claim_type", "").lower())
    except ValueError:
        return None

    subject_id = name_to_id.get(normalize_name(raw.get("subject", "")))
    object_id  = name_to_id.get(normalize_name(raw.get("object",  "")))

    if not subject_id or not object_id or subject_id == object_id:
        return None

    confidence_str = raw.get("confidence", "medium").lower()
    if confidence_str not in CONFIDENCE_GATE:
        return None

    try:
        confidence = Confidence(confidence_str)
    except ValueError:
        confidence = Confidence.MEDIUM

    excerpt = raw.get("excerpt", "")[:300]
    if not excerpt_is_grounded(excerpt, chunk.get("text", "")):
        excerpt = f"{raw.get('subject','')} {claim_type.value} {raw.get('object','')}"

    try:
        return ExtractedClaim(
            claim_id=make_claim_id(claim_type.value, subject_id, object_id),
            claim_type=claim_type,
            subject_id=subject_id,
            object_id=object_id,
            predicate_text=raw.get("predicate_text", ""),
            valid_from=raw.get("valid_from"),
            valid_until=raw.get("valid_until"),
            is_current=raw.get("valid_until") is None,
            confidence=confidence,
            evidence=[make_evidence(chunk, excerpt)],
            extraction_version=EXTRACTION_VERSION,
            extracted_at=extracted_at,
        )
    except ValidationError:
        return None


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

def extract_chunk(
    client: OpenAI,
    chunk: dict,
    extracted_at: str,
) -> tuple[list[ExtractedEntity], list[ExtractedClaim]]:

    raw_text = call_llm(client, SYSTEM_PROMPT, build_user_message(chunk))
    parsed   = parse_json(raw_text)

    if parsed is None:
        log.info("Parse failed — attempting repair")
        repaired = call_llm(client, REPAIR_PROMPT, raw_text)
        parsed   = parse_json(repaired)

    if parsed is None:
        log.warning(f"Could not parse response for {chunk['chunk_id']} — skipping")
        return [], []

    entities:     list[ExtractedEntity] = []
    name_to_id:   dict[str, str]        = {}

    for raw_ent in parsed.get("entities", [])[:8]:
        entity = build_entity(raw_ent, chunk, extracted_at)
        if entity:
            entities.append(entity)
            name_to_id[entity.canonical_name] = entity.entity_id
            for alias in entity.aliases:
                name_to_id[alias] = entity.entity_id

    claims: list[ExtractedClaim] = []
    for raw_clm in parsed.get("claims", [])[:10]:
        claim = build_claim(raw_clm, chunk, name_to_id, extracted_at)
        if claim:
            claims.append(claim)

    return entities, claims


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    return set(CHECKPOINT_PATH.read_text().splitlines())


def mark_done(chunk_id: str) -> None:
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(chunk_id + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_stats(stats: dict) -> None:
    STATS_PATH.write_text(json.dumps(stats, indent=2))


def main() -> None:
    client       = get_client()
    done         = load_checkpoint()
    extracted_at = datetime.now(timezone.utc).isoformat()

    log.info(f"Loading chunks from {CHUNKS_PATH}")
    chunks = json.loads(CHUNKS_PATH.read_text())

    chunks = chunks[START_OFFSET:]
    if MAX_CHUNKS is not None:
        chunks = chunks[:MAX_CHUNKS]

    total = len(chunks)
    log.info(f"{total} chunks to process  |  {len(done)} already done  |  model: {MODEL}")

    stats = {
        "extraction_version": EXTRACTION_VERSION,
        "model": MODEL,
        "started_at": extracted_at,
        "chunks_processed": 0,
        "entities_extracted": 0,
        "claims_extracted": 0,
        "chunks_failed": 0,
        "last_saved_at": None,
    }

    last_save_time = time.monotonic()

    with (
        open(ENTITIES_PATH,   "a") as ent_f,
        open(CLAIMS_PATH,     "a") as clm_f,
        open(QUARANTINE_PATH, "a") as qua_f,
    ):
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]

            if chunk_id in done:
                continue

            log.info(f"[{i+1}/{total}]  chunk={chunk_id}  issue={chunk.get('issue_number', '?')}")

            try:
                entities, claims = extract_chunk(client, chunk, extracted_at)
            except Exception as e:
                log.error(f"Failed on {chunk_id}: {e}")
                qua_f.write(json.dumps({"chunk_id": chunk_id, "error": str(e)}) + "\n")
                stats["chunks_failed"] += 1
                mark_done(chunk_id)
                continue

            for entity in entities:
                ent_f.write(entity.model_dump_json() + "\n")
                stats["entities_extracted"] += 1

            for claim in claims:
                clm_f.write(claim.model_dump_json() + "\n")
                stats["claims_extracted"] += 1

            mark_done(chunk_id)
            stats["chunks_processed"] += 1

            # ── periodic save every 10 minutes ──────────────────────────────
            now = time.monotonic()
            if now - last_save_time >= SAVE_INTERVAL_SECS:
                stats["last_saved_at"] = datetime.now(timezone.utc).isoformat()
                save_stats(stats)
                last_save_time = now
                log.info(
                    f"  ✓ progress saved — {stats['chunks_processed']} done | "
                    f"{stats['entities_extracted']} entities | "
                    f"{stats['claims_extracted']} claims | "
                    f"{stats['chunks_failed']} failed"
                )

            time.sleep(REQUEST_PAUSE)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_stats(stats)

    log.info("─" * 50)
    log.info(f"Done.  entities={stats['entities_extracted']}  claims={stats['claims_extracted']}  failed={stats['chunks_failed']}")
    log.info(f"Stats written to {STATS_PATH}")


if __name__ == "__main__":
    main()