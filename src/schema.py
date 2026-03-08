"""
schema.py — Ontology definitions for Layer10 memory extraction.

Entity types and claim types are designed to capture the key knowledge
units found in GitHub Issues + Comments (and extensible to email/Slack/Jira).
Every extracted object carries a mandatory evidence envelope.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import hashlib, re


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON       = "person"        # GitHub user / author
    COMPONENT    = "component"     # library module, sub-package, class, function
    BUG          = "bug"           # a reported defect
    FEATURE      = "feature"       # a requested or shipped capability
    VERSION      = "version"       # release tag / semver
    DEPENDENCY   = "dependency"    # external package / tool
    CONCEPT      = "concept"       # architectural pattern, design principle
    ORGANIZATION = "organization"  # company, team, project


class ClaimType(str, Enum):
    # Relationships between entities
    REPORTS_BUG       = "reports_bug"        # person → bug
    FIXES_BUG         = "fixes_bug"          # person/component → bug
    IMPLEMENTS        = "implements"         # person/component → feature
    REQUESTS_FEATURE  = "requests_feature"   # person → feature
    USES              = "uses"               # component → dependency
    DEPENDS_ON        = "depends_on"         # component → component
    AFFECTS           = "affects"            # bug → component
    INTRODUCED_IN     = "introduced_in"      # bug/feature → version
    FIXED_IN          = "fixed_in"           # bug → version
    MEMBER_OF         = "member_of"          # person → organization

    # State / lifecycle claims
    STATE_CHANGE      = "state_change"       # entity changed state (open→closed, etc.)
    ASSIGNED_TO       = "assigned_to"        # issue → person
    DUPLICATE_OF      = "duplicate_of"       # artifact/entity → artifact/entity

    # Factual assertions
    HAS_PROPERTY      = "has_property"       # entity has an attribute-value fact
    DECISION_MADE     = "decision_made"      # a design decision was recorded
    WORKAROUND_EXISTS = "workaround_exists"  # a workaround was documented


class Confidence(str, Enum):
    HIGH   = "high"    # explicitly stated, unambiguous
    MEDIUM = "medium"  # inferred from context
    LOW    = "low"     # speculative / hedged language in source


# ---------------------------------------------------------------------------
# Evidence envelope — required on every extracted object
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    chunk_id:    str  = Field(..., description="chunk_id from chunks.json")
    artifact_id: str  = Field(..., description="artifact_id from artifacts.json")
    excerpt:     str  = Field(..., description="Verbatim text snippet (≤ 300 chars) that supports the claim")
    char_start:  Optional[int] = Field(None, description="Character offset of excerpt start in artifact text")
    char_end:    Optional[int] = Field(None, description="Character offset of excerpt end in artifact text")
    author:      str  = Field(..., description="Author of the source artifact")
    timestamp:   str  = Field(..., description="ISO-8601 timestamp of the source artifact")
    issue_number: Optional[int] = Field(None, description="GitHub issue number for traceability")

    @field_validator("excerpt")
    @classmethod
    def truncate_excerpt(cls, v: str) -> str:
        return v[:300]


# ---------------------------------------------------------------------------
# Extracted Entity
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    entity_id:   str        = Field(..., description="Stable deterministic ID (see make_entity_id)")
    entity_type: EntityType
    canonical_name: str     = Field(..., description="Best normalized name for this entity")
    aliases:     list[str]  = Field(default_factory=list, description="Other names seen for this entity")
    description: Optional[str] = Field(None, description="Short description if available")
    confidence:  Confidence = Confidence.MEDIUM
    evidence:    list[Evidence] = Field(..., min_length=1)

    # Extraction provenance
    extraction_version: str = Field(..., description="Schema+prompt version tag, e.g. v1.0.0")
    extracted_at: str       = Field(..., description="ISO-8601 UTC timestamp of extraction run")


# ---------------------------------------------------------------------------
# Extracted Claim
# ---------------------------------------------------------------------------

class ExtractedClaim(BaseModel):
    claim_id:    str        = Field(..., description="Stable deterministic ID")
    claim_type:  ClaimType
    subject_id:  str        = Field(..., description="entity_id of the subject")
    object_id:   str        = Field(..., description="entity_id of the object")
    predicate_text: str     = Field(..., description="Human-readable statement of the claim")

    # Validity window
    valid_from:  Optional[str] = Field(None, description="ISO-8601: when this became true")
    valid_until: Optional[str] = Field(None, description="ISO-8601: when this stopped being true (null = still current)")
    is_current:  bool = True

    confidence:  Confidence = Confidence.MEDIUM
    evidence:    list[Evidence] = Field(..., min_length=1)

    extraction_version: str = Field(...)
    extracted_at: str       = Field(...)


# ---------------------------------------------------------------------------
# LLM extraction payload (what we ask the model to return per chunk)
# ---------------------------------------------------------------------------

class ChunkExtractionResult(BaseModel):
    """The structured JSON the LLM must return for a single chunk."""
    entities: list[dict] = Field(default_factory=list)
    claims:   list[dict] = Field(default_factory=list)
    # The LLM fills entity_type, canonical_name, aliases, description, confidence,
    # claim_type, subject (name), object (name), predicate_text, confidence, excerpt.
    # The pipeline resolves entity_ids and attaches Evidence envelopes.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entity_id(entity_type: str, canonical_name: str) -> str:
    """Deterministic, collision-resistant ID for an entity."""
    key = f"{entity_type.lower()}::{canonical_name.lower().strip()}"
    return "ent_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def make_claim_id(claim_type: str, subject_id: str, object_id: str) -> str:
    """Deterministic ID for a claim (type + subject + object)."""
    key = f"{claim_type}::{subject_id}::{object_id}"
    return "clm_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def normalize_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace — used before ID generation."""
    return re.sub(r"\s+", " ", name.strip().lower())