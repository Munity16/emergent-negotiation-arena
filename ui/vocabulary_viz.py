"""
ui/vocabulary_viz.py — Emergent Negotiation Arena

Renders the emergent vocabulary as an interactive D3.js force-directed
lineage tree:

    root ── context ── symbol ── agent
                                 (adopter, "adopted by")

  - context nodes  = the resource trade context a symbol dominantly means
                      (e.g. "food\u2192water")
  - symbol nodes   = an invented token, colour-coded by stability_score
                      (red = unstable/new, green = converged/stable "word")
  - agent nodes    = the 3 negotiation agents; an edge from a symbol to an
                      agent means that agent has adopted / used the symbol

Reads outputs/vocabulary.json (written by core.symbol_tracker.SymbolTracker.to_json)
and writes a single self-contained HTML file (data is embedded inline, so it
works straight from disk with no server or CORS issues).

Usage:
    python ui/vocabulary_viz.py
    python ui/vocabulary_viz.py --input outputs/vocabulary.json --output outputs/vocabulary_viz.html
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_graph(vocab_data: dict) -> dict:
    """Convert the vocabulary.json shape into a {nodes, links} graph for D3."""
    nodes = []
    links = []
    seen_context_ids = {}
    seen_agent_ids = set()

    nodes.append({
        "id": "root",
        "type": "root",
        "label": "Emergent\nVocabulary",
        "value": len(vocab_data.get("vocabulary", [])),
    })

    for entry in vocab_data.get("vocabulary", []):
        symbol = entry["symbol"]
        symbol_id = f"symbol::{symbol}"
        context = entry.get("dominant_context") or "unclassified"
        context_id = f"context::{context}"

        if context_id not in seen_context_ids:
            seen_context_ids[context_id] = True
            nodes.append({
                "id": context_id,
                "type": "context",
                "label": context,
                "value": 1,
            })
            links.append({"source": "root", "target": context_id, "kind": "groups"})

        nodes.append({
            "id": symbol_id,
            "type": "symbol",
            "label": symbol,
            "usage_count": entry.get("usage_count", 0),
            "success_rate": entry.get("success_rate", 0.0),
            "stability_score": entry.get("stability_score", 0.0),
            "first_used_by": entry.get("first_used_by", "?"),
            "first_used_round": entry.get("first_used_round", 0),
            "value": max(1, entry.get("usage_count", 1)),
        })
        links.append({"source": context_id, "target": symbol_id, "kind": "means"})

        for adopter in entry.get("adopters", []):
            agent_id = f"agent::{adopter}"
            if agent_id not in seen_agent_ids:
                seen_agent_ids.add(agent_id)
                nodes.append({
                    "id": agent_id,
                    "type": "agent",
                    "label": adopter,
                    "value": 1,
                })
            links.append({"source": symbol_id, "target": agent_id, "kind": "adopted by"})

    return {"nodes": nodes, "links": links}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Emergent Vocabulary Lineage — Negotiation Arena</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
  :root {
    --bg: #0b0b0e;
    --panel: #141418;
    --ink: #f5f3ee;
    --muted: #8a8a93;
    --grid: #26262c;
    --arena-red: #ed1c24;
    --accent-green: #3ddc84;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }
  #header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    padding: 18px 26px 14px; pointer-events: none;
    background: linear-gradient(180deg, rgba(11,11,14,0.95) 40%, rgba(11,11,14,0));
  }
  #header h1 {
    margin: 0; font-size: 20px; letter-spacing: 0.04em;
    font-family: "SFMono-Regular", Consolas, monospace; font-weight: 700;
  }
  #header h1 span { color: var(--arena-red); }
  #header p { margin: 4px 0 0; font-size: 12.5px; color: var(--muted); max-width: 640px; }
  #legend {
    position: fixed; bottom: 18px; left: 26px; z-index: 10;
    font-family: "SFMono-Regular", Consolas, monospace; font-size: 11.5px;
    color: var(--muted); background: rgba(20,20,24,0.85);
    border: 1px solid var(--grid); border-radius: 8px; padding: 10px 14px;
    line-height: 1.9;
  }
  #legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: middle; }
  #stability-scale {
    position: fixed; bottom: 18px; right: 26px; z-index: 10;
    font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px; color: var(--muted);
    background: rgba(20,20,24,0.85); border: 1px solid var(--grid); border-radius: 8px;
    padding: 10px 14px; width: 190px;
  }
  #stability-scale .bar {
    height: 8px; border-radius: 4px; margin: 6px 0;
    background: linear-gradient(90deg, #ed1c24 0%, #e0a63d 50%, #3ddc84 100%);
  }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 20;
    background: #1b1b20; border: 1px solid var(--grid); border-radius: 8px;
    padding: 10px 13px; font-size: 12.5px; color: var(--ink);
    font-family: "SFMono-Regular", Consolas, monospace;
    max-width: 260px; opacity: 0; transition: opacity 0.12s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
  }
  #tooltip .t-symbol { color: var(--accent-green); font-weight: 700; font-size: 14px; }
  #tooltip .t-row { color: var(--muted); margin-top: 3px; }
  #empty-state {
    position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
    flex-direction: column; text-align: center; color: var(--muted);
    font-family: "SFMono-Regular", Consolas, monospace;
  }
  svg { display: block; width: 100vw; height: 100vh; cursor: grab; }
  svg:active { cursor: grabbing; }
  .link { stroke-opacity: 0.55; fill: none; }
  .link-groups { stroke: var(--grid); stroke-width: 1.4; }
  .link-means { stroke: #4a4a55; stroke-width: 1.6; }
  .link-adopted { stroke: var(--accent-green); stroke-width: 1.1; stroke-dasharray: 3 3; stroke-opacity: 0.5; }
  .node-label {
    fill: var(--ink); font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 10.5px; pointer-events: none; text-anchor: middle;
  }
  .node-label-root { font-size: 12px; font-weight: 700; }
  circle.node { stroke: #0b0b0e; stroke-width: 1.6; cursor: pointer; }
</style>
</head>
<body>
  <div id="header">
    <h1>EMERGENT <span>VOCABULARY</span> LINEAGE</h1>
    <p>Every invented symbol, grouped by the trade context it converged on, and who adopted it.
       Node colour = stability score (red = unstable, green = a stable "word"). Drag to rearrange, scroll to zoom.</p>
  </div>

  <div id="legend">
    <div><span class="swatch" style="background:#5b5b66"></span>root</div>
    <div><span class="swatch" style="background:#3a3a44"></span>trade context</div>
    <div><span class="swatch" style="background:#ed1c24"></span>symbol (low stability)</div>
    <div><span class="swatch" style="background:#3ddc84"></span>symbol (high stability)</div>
    <div><span class="swatch" style="background:#2f6fed"></span>agent (adopter)</div>
  </div>

  <div id="stability-scale">
    stability score
    <div class="bar"></div>
    <div style="display:flex; justify-content:space-between;"><span>0.0</span><span>1.0</span></div>
  </div>

  <div id="tooltip"></div>
  <div id="empty-state"><div>No vocabulary yet.<br/>Run a simulation to generate outputs/vocabulary.json.</div></div>

  <script>
    const GRAPH = __GRAPH_JSON__;
    const SUMMARY = __SUMMARY_JSON__;

    const width = window.innerWidth;
    const height = window.innerHeight;

    if (!GRAPH.nodes.length || GRAPH.nodes.length === 1) {
      document.getElementById("empty-state").style.display = "flex";
    }

    const svg = d3.select("body").append("svg")
      .attr("viewBox", [0, 0, width, height]);

    const container = svg.append("g");

    svg.call(d3.zoom().scaleExtent([0.25, 4]).on("zoom", (event) => {
      container.attr("transform", event.transform);
    }));

    const colorForType = (d) => {
      if (d.type === "root") return "#5b5b66";
      if (d.type === "context") return "#3a3a44";
      if (d.type === "agent") return "#2f6fed";
      // symbol: interpolate red -> amber -> green by stability_score
      const s = d.stability_score || 0;
      return d3.interpolateRgbBasis(["#ed1c24", "#e0a63d", "#3ddc84"])(Math.min(1, s));
    };

    const radiusForNode = (d) => {
      if (d.type === "root") return 26;
      if (d.type === "context") return 14 + Math.min(10, (d.value || 1));
      if (d.type === "agent") return 16;
      return 8 + Math.min(14, (d.usage_count || 1) * 1.6);
    };

    const linkClass = (d) => {
      if (d.kind === "groups") return "link link-groups";
      if (d.kind === "means") return "link link-means";
      return "link link-adopted";
    };

    const simulation = d3.forceSimulation(GRAPH.nodes)
      .force("link", d3.forceLink(GRAPH.links).id(d => d.id).distance(d => {
        if (d.kind === "groups") return 130;
        if (d.kind === "means") return 90;
        return 60;
      }).strength(0.7))
      .force("charge", d3.forceManyBody().strength(-260))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => radiusForNode(d) + 14));

    const link = container.append("g")
      .selectAll("line")
      .data(GRAPH.links)
      .join("line")
      .attr("class", linkClass);

    const node = container.append("g")
      .selectAll("g")
      .data(GRAPH.nodes)
      .join("g")
      .call(drag(simulation));

    node.append("circle")
      .attr("class", "node")
      .attr("r", radiusForNode)
      .attr("fill", colorForType);

    node.append("text")
      .attr("class", d => "node-label" + (d.type === "root" ? " node-label-root" : ""))
      .attr("dy", d => radiusForNode(d) + 13)
      .text(d => d.label);

    const tooltip = d3.select("#tooltip");

    node.on("mouseenter", (event, d) => {
      let html = `<div class="t-symbol">${d.label}</div>`;
      if (d.type === "symbol") {
        html += `<div class="t-row">usage: ${d.usage_count} \u00b7 success: ${(d.success_rate*100).toFixed(0)}%</div>`;
        html += `<div class="t-row">stability: ${d.stability_score.toFixed(3)}</div>`;
        html += `<div class="t-row">first used by ${d.first_used_by} (round ${d.first_used_round})</div>`;
      } else if (d.type === "context") {
        html += `<div class="t-row">trade context cluster</div>`;
      } else if (d.type === "agent") {
        html += `<div class="t-row">negotiation agent</div>`;
      } else {
        html += `<div class="t-row">${SUMMARY.total_symbols || 0} total symbols \u00b7 ${SUMMARY.stable_words || 0} stable words</div>`;
      }
      tooltip.html(html).style("opacity", 1);
    })
    .on("mousemove", (event) => {
      tooltip.style("left", (event.clientX + 16) + "px").style("top", (event.clientY + 12) + "px");
    })
    .on("mouseleave", () => tooltip.style("opacity", 0));

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function drag(sim) {
      function dragstarted(event, d) {
        if (!event.active) sim.alphaTarget(0.25).restart();
        d.fx = d.x; d.fy = d.y;
      }
      function dragged(event, d) {
        d.fx = event.x; d.fy = event.y;
      }
      function dragended(event, d) {
        if (!event.active) sim.alphaTarget(0);
        d.fx = null; d.fy = null;
      }
      return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
    }
  </script>
</body>
</html>
"""


def generate_vocab_html(input_path: str, output_path: str) -> str:
    with open(input_path) as f:
        vocab_data = json.load(f)

    graph = _build_graph(vocab_data)
    summary = vocab_data.get("summary", {})

    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(graph))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summary))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render the emergent vocabulary as a D3 lineage tree HTML page")
    parser.add_argument("--input", type=str, default="outputs/vocabulary.json",
                        help="Path to vocabulary.json (default: outputs/vocabulary.json)")
    parser.add_argument("--output", type=str, default="outputs/vocabulary_viz.html",
                        help="Output HTML path (default: outputs/vocabulary_viz.html)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"No vocabulary file at {args.input} — writing an empty-state page instead.")
        os.makedirs(os.path.dirname(args.input) or ".", exist_ok=True)
        with open(args.input, "w") as f:
            json.dump({"summary": {}, "vocabulary": [], "convergence_events": [], "extinction_events": []}, f)
    out = generate_vocab_html(args.input, args.output)
    print(f"Vocabulary lineage tree saved to: {out}")


if __name__ == "__main__":
    main()
