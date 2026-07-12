"""
ui/benchmark_card.py — Emergent Negotiation Arena

Generates a shareable PNG "benchmark card": a bar chart comparing
sequential vs. parallel multi-agent LLM inference, with a
big "X x speedup" headline — sized and styled for posting on X and
sharing.

Reads whichever benchmark JSON is available:
  outputs/benchmark_llm.json  (from core/benchmark.py)
  outputs/benchmark.json      (from a full core.simulation run)

Usage:
    python ui/benchmark_card.py
    python ui/benchmark_card.py --input outputs/benchmark_llm.json --output outputs/benchmark_card.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── High-contrast dev-tool palette ──
BG = "#0B0B0E"
PANEL = "#141418"
INK = "#F5F3EE"
MUTED = "#8A8A93"
ARENA_RED = "#ED1C24"
ACCENT_GREEN = "#3DDC84"
GRID_LINE = "#26262C"


def _load_benchmark(input_path: str) -> dict:
    with open(input_path) as f:
        return json.load(f)


def _extract_metrics(data: dict) -> dict:
    """
    Normalises either benchmark_llm.json (core/benchmark.py) or
    benchmark.json (core/simulation.py BenchmarkRecorder) into a
    common shape.
    """
    if "before_after" in data:
        ba = data["before_after"]
        return {
            "sequential_ms": ba["before_sequential_avg_ms"],
            "parallel_ms": ba["after_parallel_avg_ms"],
            "speedup": ba["speedup_factor"],
            "rounds": data.get("meta", {}).get("rounds", len(data.get("per_round", []))),
            "mode": data.get("meta", {}).get("mode", "measured"),
            "model": data.get("meta", {}).get("model", "LLM agents"),
        }
    elif "summary" in data:
        s = data["summary"]
        return {
            "sequential_ms": s["avg_sequential_estimate_ms"],
            "parallel_ms": s["avg_parallel_inference_ms"],
            "speedup": s["parallel_speedup_factor"],
            "rounds": s.get("total_rounds", 0),
            "mode": "simulation",
            "model": "LLM agents",
        }
    else:
        raise ValueError("Unrecognised benchmark JSON shape — expected "
                          "'before_after' or 'summary' key.")


def generate_benchmark_card(input_path: str, output_path: str) -> str:
    data = _load_benchmark(input_path)
    metrics = _extract_metrics(data)

    sequential_ms = metrics["sequential_ms"]
    parallel_ms = metrics["parallel_ms"]
    speedup = metrics["speedup"]

    fig = plt.figure(figsize=(10, 6.25), dpi=200, facecolor=BG)

    # Overall layout: headline block on top, bar chart below, footer strip
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 2.1, 0.4], hspace=0.35)

    # ── Headline ──
    ax_head = fig.add_subplot(gs[0])
    ax_head.axis("off")
    ax_head.set_facecolor(BG)

    ax_head.text(
        0.0, 0.78, "E M E R G E N T   N E G O T I A T I O N   A R E N A",
        transform=ax_head.transAxes, fontsize=12, color=MUTED,
        family="monospace", fontweight="bold", ha="left", va="top",
    )
    ax_head.text(
        0.0, 0.52, f"{speedup:.2f}\u00d7 SPEEDUP",
        transform=ax_head.transAxes, fontsize=44, color=ARENA_RED,
        fontweight="black", ha="left", va="top", family="sans-serif",
    )
    ax_head.text(
        0.0, 0.08,
        f"3-agent parallel LLM inference  \u00b7  {metrics['model']}",
        transform=ax_head.transAxes, fontsize=12.5, color=INK,
        ha="left", va="top", family="sans-serif",
    )

    # ── Bar chart ──
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(PANEL)

    labels = ["Sequential\n(3 calls, one-at-a-time)", "Parallel\n(asyncio.gather)"]
    values = [sequential_ms, parallel_ms]
    colors = [MUTED, ARENA_RED]

    bar_x = [0, 1]
    bars = ax.bar(bar_x, values, width=0.55, color=colors, zorder=3,
                  edgecolor="none")

    for x, v, c in zip(bar_x, values, colors):
        ax.text(x, v + max(values) * 0.03, f"{v:.0f} ms",
                ha="center", va="bottom", fontsize=17, color=INK,
                fontweight="bold", family="sans-serif")

    # Speedup connector annotation
    ax.annotate(
        "", xy=(1, parallel_ms), xytext=(0, sequential_ms),
        arrowprops=dict(arrowstyle="-", color=ACCENT_GREEN, lw=1.4,
                        linestyle=(0, (4, 3))),
    )

    ax.set_xticks(bar_x)
    ax.set_xticklabels(labels, color=INK, fontsize=11.5, family="sans-serif")
    ax.set_ylabel("avg ms / round", color=MUTED, fontsize=10.5, family="monospace")
    ax.tick_params(axis="y", colors=MUTED, labelsize=9.5)
    ax.tick_params(axis="x", colors=INK, length=0)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_LINE)
    ax.yaxis.grid(True, color=GRID_LINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.28)

    mode = metrics.get("mode", "measured")
    if mode == "simulate":
        ax.text(
            0.985, 0.94, "SYNTHETIC TIMING MODEL\n(re-run --mode live with API key)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=MUTED, family="monospace", style="italic",
        )

    # ── Footer ──
    ax_foot = fig.add_subplot(gs[2])
    ax_foot.axis("off")
    ax_foot.set_facecolor(BG)
    ax_foot.text(
        0.0, 0.5, f"{metrics['rounds']} rounds benchmarked  \u00b7  Qwen3-1.7B \u00d7 3 agents",
        transform=ax_foot.transAxes, fontsize=9.5, color=MUTED,
        family="monospace", ha="left", va="center",
    )
    ax_foot.text(
        1.0, 0.5, "Emergent Negotiation Arena",
        transform=ax_foot.transAxes, fontsize=9.5, color=ARENA_RED,
        family="monospace", fontweight="bold", ha="right", va="center",
    )

    fig.patch.set_facecolor(BG)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, facecolor=BG, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a shareable benchmark PNG card")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to benchmark JSON (default: auto-detect in outputs/)")
    parser.add_argument("--output", type=str, default="outputs/benchmark_card.png",
                        help="Output PNG path (default: outputs/benchmark_card.png)")
    return parser.parse_args()


def _autodetect_input() -> str:
    candidates = [
        "outputs/benchmark_llm.json",
        "benchmark_llm.json",
        "outputs/benchmark.json",
        "benchmark.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No benchmark JSON found. Run core/benchmark.py or core/simulation.py first, "
        "or pass --input explicitly."
    )


def main():
    args = parse_args()
    input_path = args.input or _autodetect_input()
    out = generate_benchmark_card(input_path, args.output)
    print(f"Benchmark card saved to: {out}")


if __name__ == "__main__":
    main()
