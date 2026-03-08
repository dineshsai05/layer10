"""
visualization.py

Builds a NetworkX graph from the memory store and renders it as a
self-contained interactive HTML using pyvis (works fully offline).

Reads:   data/processed/canonical_entities.json
         data/processed/canonical_claims.json
         data/processed/merge_log.json
Writes:  data/processed/viz/index.html

Install deps:
    pip install networkx pyvis

Run from project root:
    python src/visualization.py

Then open in browser:
    xdg-open data/processed/viz/index.html
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENTITIES_PATH  = Path("data/processed/canonical_entities.json")
CLAIMS_PATH    = Path("data/processed/canonical_claims.json")
MERGE_LOG_PATH = Path("data/processed/merge_log.json")
OUTPUT_DIR     = Path("data/processed/viz")
OUTPUT_HTML    = OUTPUT_DIR / "index.html"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MAX_NODES = 150

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colors per entity type
# ---------------------------------------------------------------------------

COLORS = {
    "person":       "#4f9cf9",
    "component":    "#f97316",
    "bug":          "#ef4444",
    "feature":      "#22c55e",
    "version":      "#a855f7",
    "dependency":   "#eab308",
    "concept":      "#06b6d4",
    "organization": "#ec4899",
}

# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------

def node_tooltip(entity: dict, merge_lookup: dict) -> str:
    name       = entity.get("canonical_name", "")
    etype      = entity.get("entity_type", "")
    confidence = entity.get("confidence", "")
    aliases    = entity.get("aliases", [])
    evidence   = entity.get("evidence", [])
    desc       = entity.get("description", "")
    merge_info = merge_lookup.get(entity["entity_id"])

    lines = [f"<b>{name}</b>", f"Type: {etype} | Confidence: {confidence}"]

    if desc:
        lines.append(f"<i>{textwrap.shorten(desc, 80)}</i>")

    if aliases:
        lines.append(f"Aliases: {', '.join(aliases[:5])}")

    if merge_info:
        n_merged = len(merge_info.get("merged_ids", []))
        lines.append(f"Merged from {n_merged} records ({merge_info['merge_type']})")

    if evidence:
        lines.append(f"Evidence ({len(evidence)}):")
        for ev in evidence[:2]:
            excerpt   = (ev.get("excerpt") or "")[:80]
            author    = ev.get("author", "?")
            issue     = ev.get("issue_number")
            issue_str = f"#{issue}" if issue else "?"
            date      = (ev.get("timestamp") or "")[:10]
            lines.append(f'  "{excerpt}"')
            lines.append(f"  -- {author}, issue {issue_str}, {date}")

    return "<br>".join(lines)


def edge_tooltip(claim: dict, entity_lookup: dict) -> str:
    subj_name = entity_lookup.get(claim["subject_id"], {}).get("canonical_name", "?")
    obj_name  = entity_lookup.get(claim["object_id"],  {}).get("canonical_name", "?")
    ctype     = claim.get("claim_type", "")
    predicate = claim.get("predicate_text", "")
    conf      = claim.get("confidence", "")
    current   = "current" if claim.get("is_current") else "historical"
    evidence  = claim.get("evidence", [])

    lines = [
        f"<b>{subj_name} -> {obj_name}</b>",
        f"Type: {ctype} | {conf} | {current}",
        f"{textwrap.shorten(predicate, 100)}",
    ]

    if evidence:
        lines.append(f"Evidence ({len(evidence)}):")
        for ev in evidence[:2]:
            excerpt   = (ev.get("excerpt") or "")[:80]
            author    = ev.get("author", "?")
            issue     = ev.get("issue_number")
            issue_str = f"#{issue}" if issue else "?"
            date      = (ev.get("timestamp") or "")[:10]
            lines.append(f'  "{excerpt}"')
            lines.append(f"  -- {author}, issue {issue_str}, {date}")

    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    entities  = json.loads(ENTITIES_PATH.read_text())
    claims    = json.loads(CLAIMS_PATH.read_text())
    merge_log = json.loads(MERGE_LOG_PATH.read_text()) if MERGE_LOG_PATH.exists() else {}

    merge_lookup  = {e["canonical_id"]: e for e in merge_log.get("entity_merges", [])}
    entity_lookup = {e["entity_id"]: e for e in entities}

    # Top N entities by evidence count
    entities_sorted = sorted(entities, key=lambda e: len(e.get("evidence", [])), reverse=True)
    top_entities    = entities_sorted[:MAX_NODES]
    top_entity_ids  = {e["entity_id"] for e in top_entities}
    log.info(f"Using top {len(top_entities)} entities")

    # Build NetworkX DiGraph
    G = nx.DiGraph()

    for entity in top_entities:
        eid      = entity["entity_id"]
        etype    = entity.get("entity_type", "concept")
        ev_count = len(entity.get("evidence", []))
        size     = max(10, min(40, 10 + ev_count * 4))
        color    = COLORS.get(etype, "#94a3b8")
        label    = entity["canonical_name"][:25]
        title    = node_tooltip(entity, merge_lookup)

        G.add_node(eid, label=label, title=title, color=color, size=size, group=etype)

    seen_pairs = set()
    for claim in claims:
        subj = claim["subject_id"]
        obj  = claim["object_id"]
        if subj not in top_entity_ids or obj not in top_entity_ids:
            continue
        pair = (subj, obj)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        G.add_edge(
            subj, obj,
            title=edge_tooltip(claim, entity_lookup),
            label=claim["claim_type"].replace("_", " "),
            color="#475569",
        )

    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Render with pyvis
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0a0f1a",
        font_color="#e2e8f0",
        directed=True,
        notebook=False,
    )
    net.from_nx(G)

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.1,
          "springLength": 140,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": { "enabled": true, "iterations": 300 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "font": { "size": 9, "color": "#64748b" },
        "width": 1,
        "selectionWidth": 2,
        "arrows": { "to": { "enabled": true } }
      },
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "font": { "size": 11 }
      }
    }
    """)

    net.save_graph(str(OUTPUT_HTML))

    # Inject legend overlay
    inject_ui(OUTPUT_HTML, entities, claims, merge_log)

    log.info(f"Done -> {OUTPUT_HTML}")
    log.info(f"Open:  xdg-open {OUTPUT_HTML.resolve()}")


# ---------------------------------------------------------------------------
# Inject legend + stats bar
# ---------------------------------------------------------------------------

def inject_ui(html_path: Path, entities: list, claims: list, merge_log: dict) -> None:
    html = html_path.read_text(encoding="utf-8")

    type_counts: dict[str, int] = {}
    for e in entities:
        t = e.get("entity_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    entity_merges = len(merge_log.get("entity_merges", []))

    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;'
        f'font-size:11px;color:#94a3b8;font-family:monospace">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{COLORS.get(t,"#94a3b8")};'
        f'display:inline-block"></span>{t} ({n})</span>'
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1])
    )

    overlay = f"""
    <style>
      body {{ margin: 0; overflow: hidden; }}
      #layer10-header {{
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
        background: rgba(17,24,39,0.95); border-bottom: 1px solid #1e3a5f;
        padding: 8px 16px;
        display: flex; align-items: center; justify-content: space-between;
        font-family: monospace;
      }}
      .hstat {{ text-align: center; margin-left: 20px; }}
      .hstat-v {{ color: #3b82f6; font-size: 16px; font-weight: 600; }}
      .hstat-l {{ color: #64748b; font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; }}
      #layer10-legend {{
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000;
        background: rgba(17,24,39,0.95); border-top: 1px solid #1e3a5f;
        padding: 8px 16px;
        display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
      }}
      #mynetwork {{
        margin-top: 48px;
        height: calc(100vh - 88px) !important;
      }}
    </style>

    <div id="layer10-header">
      <span style="color:#06b6d4;font-size:13px;font-weight:600;letter-spacing:0.08em">
        LAYER10 · MEMORY GRAPH
      </span>
      <div style="display:flex;align-items:center">
        <div class="hstat"><div class="hstat-v">{len(entities)}</div><div class="hstat-l">Entities</div></div>
        <div class="hstat"><div class="hstat-v">{len(claims)}</div><div class="hstat-l">Claims</div></div>
        <div class="hstat"><div class="hstat-v">{entity_merges}</div><div class="hstat-l">Merges</div></div>
        <div class="hstat" style="margin-left:16px;font-size:10px;color:#64748b;text-align:right">
          Hover over nodes/edges<br>to see evidence
        </div>
      </div>
    </div>

    <div id="layer10-legend">{legend_items}</div>
    """

    html = html.replace("</body>", overlay + "\n</body>")
    html_path.write_text(html, encoding="utf-8")
    log.info("UI overlay injected")


if __name__ == "__main__":
    main()