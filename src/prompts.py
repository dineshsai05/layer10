"""
prompts.py — Prompt templates for structured extraction.

Design principles:
  1. Explicit schema in the prompt — the model knows exactly what JSON to emit.
  2. Evidence-first instruction — the model must quote source text.
  3. Confidence vocabulary is defined inline — reduces hallucinated values.
  4. Negative examples discourage hallucination and over-extraction.
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are a structured knowledge extraction engine for a long-term memory system.
Your job is to read a GitHub issue chunk and extract typed entities and claims
that represent durable, grounded knowledge.

## Output format
Return ONLY a valid JSON object with this exact structure — no markdown, no prose:

{
  "entities": [
    {
      "canonical_name": "string — best normalized name",
      "entity_type": "person | component | bug | feature | version | dependency | concept | organization",
      "aliases": ["other names seen in this chunk"],
      "description": "one-sentence description or null",
      "confidence": "high | medium | low",
      "excerpt": "verbatim quote from the chunk (≤ 200 chars) that establishes this entity"
    }
  ],
  "claims": [
    {
      "claim_type": "reports_bug | fixes_bug | implements | requests_feature | uses | depends_on | affects | introduced_in | fixed_in | member_of | state_change | assigned_to | duplicate_of | has_property | decision_made | workaround_exists",
      "subject": "canonical_name of subject entity",
      "object": "canonical_name of object entity",
      "predicate_text": "plain-English statement of the claim",
      "confidence": "high | medium | low",
      "excerpt": "verbatim quote from the chunk (≤ 200 chars) that supports this claim",
      "valid_from": "ISO date or null",
      "valid_until": "ISO date or null"
    }
  ]
}

## Entity extraction rules
- Only extract entities that are explicitly named or described in the chunk.
- "person" = GitHub username or real name that appears as an actor.
- "component" = a named module, class, function, or subsystem of LangChain or a dependency.
- "bug" = a described defect, error, or unintended behaviour — give it a descriptive name.
- "feature" = a named capability, option, or enhancement being discussed.
- "version" = a semver string like "0.1.4" or release name like "v0.2".
- "dependency" = an external package like "openai", "tiktoken", "faiss-cpu".
- "concept" = an architectural idea, pattern, or term being defined.
- "organization" = a company, team, or project name.
- Use confidence="high" only when the entity is unambiguously named.
- Do NOT invent entities not present in the text.

## Claim extraction rules
- Every claim must have both a subject and object that are entities you extracted (or entities well-established by the issue title/context provided).
- The excerpt must be a verbatim substring from the chunk text.
- Use valid_from / valid_until only when the chunk explicitly mentions dates or version ranges.
- For state_change claims, subject = the issue/bug entity, predicate_text = "changed state from X to Y".
- Do NOT fabricate relationships. If you are not sure, use confidence="low" or omit the claim.
- Maximum 8 entities and 10 claims per chunk.

## Negative examples (do NOT do these)
- Do not add an entity just because a word sounds technical.
- Do not create a claim with subject == object.
- Do not use an excerpt that is not a substring of the provided chunk text.
- Do not emit markdown code fences or explanation text outside the JSON.
"""


def build_extraction_user_message(chunk: dict) -> str:
    """
    Build the user-turn message for a single chunk.
    chunk: a record from chunks.json
    """
    return f"""\
## Chunk metadata
- chunk_id: {chunk["chunk_id"]}
- artifact_id: {chunk["artifact_id"]}
- issue_title: {chunk.get("issue_title", "unknown")}
- issue_state: {chunk.get("issue_state", "unknown")}
- author: {chunk.get("author", "unknown")}
- timestamp: {chunk.get("timestamp", "unknown")}

## Chunk text
{chunk["text"]}

Extract entities and claims from the chunk text above.
Remember: excerpts must be verbatim substrings of the chunk text.
Return only JSON.
"""


REPAIR_SYSTEM_PROMPT = """\
You are a JSON repair assistant.
You will be given a malformed or incomplete JSON string that should conform to this schema:

{
  "entities": [...],
  "claims": [...]
}

Fix any syntax errors, missing brackets, or invalid field values.
- Allowed entity_type values: person, component, bug, feature, version, dependency, concept, organization
- Allowed claim_type values: reports_bug, fixes_bug, implements, requests_feature, uses, depends_on,
  affects, introduced_in, fixed_in, member_of, state_change, assigned_to, duplicate_of,
  has_property, decision_made, workaround_exists
- Allowed confidence values: high, medium, low
Return ONLY the corrected JSON object. No markdown. No explanation.
"""


def build_repair_user_message(bad_json: str) -> str:
    return f"Fix this JSON:\n\n{bad_json}"