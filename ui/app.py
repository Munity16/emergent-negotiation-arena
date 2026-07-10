"""
app.py — Emergent Negotiation Arena — Gradio dashboard

Terminal-styled live dashboard showing:
  - Backend health panel (Fireworks / replay / heuristic, active mode, last error)
  - World map (10×10 grid: resource zones, tiles, agents)
  - Agent cards with segmented resource meters, scores, trade stats
  - Trade activity chart (per-round bars)
  - Vocabulary table (invented symbols, stability, adopters)
  - Convergence feed (when a "word" is born)
  - Benchmark tiles: parallel vs sequential inference time
  - Trade log

Run:
  python ui/app.py
"""

import asyncio
import html as html_lib
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.grid_world import GRID_SIZE, RESOURCES
from core.simulation import Simulation
from core.replay import ScriptedAgentPool, load_replay_file
from agents.llm_agent import (
    BackendUnavailable,
    check_fireworks_ready,
    resolve_backend,
)

REPLAY_FILE = os.getenv("REPLAY_FILE", "outputs/sample_run.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")


# ─────────────────────────────────────────────
# Global simulation state
#
# The simulation runs in a background thread while Gradio polls from the UI
# thread. Rules that keep this safe:
#   - _sim_lock guards every read/write of the module globals below.
#   - Display code never iterates live simulation structures; it reads only
#     sim.latest_snapshot, an immutable dict the simulation swaps in
#     atomically at the end of each round.
# ─────────────────────────────────────────────

_sim: Simulation | None = None
_sim_thread: threading.Thread | None = None
_sim_lock = threading.Lock()
_running = False
_last_error: str | None = None
_active_backend: str | None = None
_backend_reason: str = ""

# Health probes are rate-limited so a dashboard refresh doesn't hammer
# the checks on every poll tick.
_probe_cache = {"t": 0.0, "fireworks": (False, "")}
_PROBE_TTL_S = 8.0


async def _run_and_close(sim: Simulation):
    try:
        await sim.run()
    finally:
        await sim.agent_pool.close()


def _run_simulation_thread(num_rounds: int, backend: str, seed: int, semantics: str):
    """Runs the simulation in a background thread; never lets an exception die silently."""
    global _sim, _running, _last_error
    try:
        os.environ["AGENT_BACKEND"] = backend
        os.environ["SEMANTICS_MODE"] = semantics

        agent_pool = None
        rounds, run_seed, out_dir = num_rounds, seed, OUTPUT_DIR
        if backend == "replay":
            meta, recorded = load_replay_file(REPLAY_FILE)
            agent_pool = ScriptedAgentPool(recorded, playback_delay_s=0.35)
            rounds = len(recorded)
            run_seed = meta.get("seed", seed)
            out_dir = os.path.join(OUTPUT_DIR, "replay")

        sim = Simulation(num_rounds=rounds, seed=run_seed, verbose=False,
                         output_dir=out_dir, agent_pool=agent_pool)
        with _sim_lock:
            _sim = sim

        asyncio.run(_run_and_close(sim))
        sim.save_outputs()
    except Exception as e:
        with _sim_lock:
            _last_error = f"{type(e).__name__}: {e}"
    finally:
        with _sim_lock:
            _running = False


# ─────────────────────────────────────────────
# Design tokens — terminal/LCD theme, single committed dark look.
#
# Resource "fire ramp" and agent accents were run through the dataviz
# palette validator on the #0a0a0a surface: CVD separation, chroma and
# contrast PASS; the lightness band is deliberately exceeded on the bright
# end — the blazing-on-black ramp IS the design — and every colored mark
# carries a visible text label (relief rule).
# ─────────────────────────────────────────────

RESOURCE_LETTER = {"food": "f", "water": "w", "metal": "m", "energy": "e"}
# Dark ink on the two bright fire steps, light ink on the deep ones.
RESOURCE_INK = {"food": "#ffffff", "water": "#ffffff",
                "metal": "#0a0a0a", "energy": "#0a0a0a"}
AGENT_IDS = ["agent_0", "agent_1", "agent_2"]

# Loaded via Blocks(head=...): gradio concatenates its own CSS ahead of ours
# in one <style> tag, which makes a CSS @import invalid and silently dropped.
FONT_HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?family=VT323&display=swap">'
)

CUSTOM_CSS = """
:root, .dark, .gradio-container {
  --ena-page: #0a0a0a;
  --ena-surface: #0d0d0d;
  --ena-ink: #f5f5f5;
  --ena-ink2: #b3b3b3;
  --ena-muted: #6f6f6f;
  --ena-hairline: #222222;
  --ena-ring: #2b2b2b;
  --ena-res-food: #b91c1c;
  --ena-res-water: #ea580c;
  --ena-res-metal: #f5a623;
  --ena-res-energy: #ffee58;
  --ena-agent-0: #34d399;
  --ena-agent-1: #1d9bf0;
  --ena-agent-2: #f472b6;
  --ena-good: #34d399;
  --ena-serious: #f5a623;
  --ena-critical: #ff5555;
  --ena-chipbg: rgba(255,255,255,0.05);
  --ena-mono: 'IBM Plex Mono', ui-monospace, Consolas, 'Courier New', monospace;
  --ena-display: 'VT323', 'IBM Plex Mono', ui-monospace, monospace;
}

body, .gradio-container, .dark .gradio-container {
  background: var(--ena-page) !important;
  font-family: var(--ena-mono) !important;
}
.gradio-container * { font-family: var(--ena-mono); }

/* Gradio chrome → terminal */
.gradio-container .block, .gradio-container .form,
.gradio-container fieldset, .gradio-container .panel {
  background: transparent !important;
  border-color: var(--ena-ring) !important;
}
.gradio-container label span, .gradio-container .block-title,
.gradio-container label > span[data-testid="block-info"] {
  color: var(--ena-ink2) !important;
  background: transparent !important;
  font-family: var(--ena-mono) !important;
  text-transform: uppercase;
  font-size: 11px !important;
  letter-spacing: .08em;
}
.gradio-container .wrap .info, .gradio-container span.info,
.gradio-container .block .info { color: var(--ena-muted) !important; font-size: 11px !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: #111 !important; color: var(--ena-ink) !important;
  border: 1px solid var(--ena-ring) !important;
  font-family: var(--ena-mono) !important;
}
.gradio-container button.primary {
  background: #f5f5f5 !important; color: #0a0a0a !important;
  border: 1px solid #f5f5f5 !important; border-radius: 2px !important;
  font-family: var(--ena-mono) !important; font-weight: 600 !important;
}
.gradio-container button.primary:hover { background: #ffffff !important; }
.gradio-container button.secondary {
  background: transparent !important; color: var(--ena-ink) !important;
  border: 1px dashed var(--ena-ring) !important; border-radius: 2px !important;
  font-family: var(--ena-mono) !important;
}
/* Radio pills — scoped to fieldset so block labels (e.g. the slider's
   "Number of rounds") don't inherit the pill background */
.gradio-container fieldset .wrap label {
  background: #111 !important; border: 1px solid var(--ena-ring) !important;
  border-radius: 2px !important; color: var(--ena-ink2) !important;
}
.gradio-container fieldset .wrap label.selected {
  background: #f5f5f5 !important; color: #0a0a0a !important;
}

/* The dashboard re-polls every ~2.5s; gradio dims/fades output components
   while an event is pending, which reads as constant flicker. Pin them. */
.gradio-container .pending { opacity: 1 !important; }
.gradio-container .generating { border: none !important; }

/* ── Panels ── */
.ena-card {
  background: var(--ena-surface);
  border: 1px solid var(--ena-ring);
  border-radius: 2px;
  padding: 14px 16px;
  font-family: var(--ena-mono);
  color: var(--ena-ink);
}
.ena-deco {
  display: flex; justify-content: space-between; align-items: center;
  color: var(--ena-muted); font-size: 10px; letter-spacing: 2px;
  margin-bottom: 10px; user-select: none;
}
.ena-deco .sq { color: #3a3a3a; }
.ena-card-head {
  font-size: 15px; font-weight: 400; color: var(--ena-ink);
  font-family: var(--ena-display); font-size: 22px; letter-spacing: .04em;
  text-transform: uppercase;
  display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px;
}
.ena-sub {
  font-size: 11px; font-weight: 400; color: var(--ena-muted);
  font-family: var(--ena-mono); text-transform: none; letter-spacing: 0;
}
.ena-empty { color: var(--ena-muted); font-size: 12px; padding: 8px 0; }

/* Bracket chip: [ content ] */
.ena-bracket { white-space: nowrap; }
.ena-bracket::before { content: "[ "; color: var(--ena-muted); }
.ena-bracket::after  { content: " ]"; color: var(--ena-muted); }

/* ── World map ── */
.ena-grid {
  display: grid;
  grid-template-columns: repeat(10, 30px);
  grid-auto-rows: 30px;
  gap: 1px;
  width: max-content;
  background: var(--ena-hairline);
  border: 1px solid var(--ena-hairline);
}
.ena-cell {
  position: relative;
  display: flex; align-items: center; justify-content: center;
  background: var(--ena-page);
}
.ena-dot {
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 700; line-height: 1;
}
.ena-dot-empty {
  width: 10px; height: 10px; border-radius: 50%;
  border: 2px dashed #3a3a3a; opacity: .8;
}
.ena-agent-badge {
  border-radius: 2px;
  display: flex; align-items: center; justify-content: center;
  color: #0a0a0a; font-weight: 700;
  box-shadow: 0 0 0 2px var(--ena-page);
  z-index: 2;
}
.ena-cell .ena-corner {
  position: absolute; right: 2px; bottom: 2px;
  width: 9px; height: 9px; font-size: 0;
}
.ena-legend {
  display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 12px;
  font-size: 11px; color: var(--ena-ink2); align-items: center;
}
.ena-legend .chip { display: inline-flex; align-items: center; gap: 6px; }
.ena-swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.ena-swatch-sq { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

/* ── Agent cards ── */
.ena-cards { display: flex; gap: 10px; flex-wrap: wrap; }
.ena-agent-card {
  flex: 1 1 220px; min-width: 220px;
  background: var(--ena-surface);
  border: 1px solid var(--ena-ring); border-radius: 2px; padding: 12px 14px;
  font-family: var(--ena-mono);
}
.ena-agent-card.dead { opacity: .55; }
.ena-agent-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.ena-agent-head .name { font-weight: 600; font-size: 12px; color: var(--ena-ink); }
.ena-pill { margin-left: auto; font-size: 10px; font-weight: 600; letter-spacing: .05em; }
.ena-pill::before { content: "[ "; color: var(--ena-muted); font-weight: 400; }
.ena-pill::after  { content: " ]"; color: var(--ena-muted); font-weight: 400; }
.ena-pill.alive { color: var(--ena-good); }
.ena-pill.dead-pill { color: var(--ena-critical); }
.ena-score-row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; }
.ena-score {
  font-size: 30px; font-weight: 400; color: var(--ena-ink);
  font-family: var(--ena-display); line-height: 1;
}
.ena-score-lbl { font-size: 10px; color: var(--ena-muted); text-transform: uppercase; }
.ena-trades-lbl { margin-left: auto; font-size: 11px; color: var(--ena-ink2); }
.ena-bar-row {
  display: grid; grid-template-columns: 58px 1fr 30px;
  align-items: center; gap: 8px; margin: 5px 0; font-size: 11px;
}
.ena-bar-row .lbl { color: var(--ena-ink2); }
.ena-bar-row .val { color: var(--ena-ink); text-align: right; font-variant-numeric: tabular-nums; }
.ena-track { height: 8px; background: #161616; border: 1px solid var(--ena-hairline); overflow: hidden; }
.ena-fill {
  height: 100%;
  /* segmented LCD meter */
  -webkit-mask: repeating-linear-gradient(90deg, #000 0 4px, transparent 4px 6px);
          mask: repeating-linear-gradient(90deg, #000 0 4px, transparent 4px 6px);
}
.ena-low { color: var(--ena-critical); font-weight: 700; font-size: 9px; margin-left: 4px; }

/* ── Activity chart ── */
.ena-chart {
  display: flex; align-items: flex-end; gap: 2px;
  height: 120px; padding: 6px 2px 2px 2px;
  border: 1px solid var(--ena-hairline);
  background:
    repeating-linear-gradient(0deg, var(--ena-hairline) 0 1px, transparent 1px 30px),
    var(--ena-page);
}
.ena-chart .bar {
  flex: 1 1 auto; min-width: 3px; max-width: 14px;
  display: flex; flex-direction: column-reverse;
}
.ena-chart .seg-ok   { background: var(--ena-res-energy); }
.ena-chart .seg-miss { background: var(--ena-res-food); }
.ena-chart .seg-none { background: #2a2a2a; height: 2px; }
.ena-axis {
  display: flex; justify-content: space-between; margin-top: 4px;
  color: var(--ena-muted); font-size: 10px; font-variant-numeric: tabular-nums;
}

/* ── Vocabulary table ── */
.ena-table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--ena-mono); }
.ena-table th {
  text-align: left; font-size: 10px; font-weight: 600; color: var(--ena-muted);
  text-transform: uppercase; letter-spacing: .08em;
  padding: 6px 10px 6px 0; border-bottom: 1px solid var(--ena-ring);
}
.ena-table td {
  padding: 7px 10px 7px 0; border-bottom: 1px solid var(--ena-hairline);
  color: var(--ena-ink); vertical-align: middle;
  font-variant-numeric: tabular-nums;
}
.ena-sym {
  font-family: var(--ena-mono); font-size: 11.5px; font-weight: 700;
  color: var(--ena-ink); white-space: nowrap;
}
.ena-sym::before { content: "[ "; color: var(--ena-muted); font-weight: 400; }
.ena-sym::after  { content: " ]"; color: var(--ena-muted); font-weight: 400; }
.ena-stab { display: flex; align-items: center; gap: 8px; min-width: 110px; }
.ena-stab .ena-track { flex: 1; }
.ena-adopter {
  width: 15px; height: 15px; border-radius: 2px; display: inline-flex;
  align-items: center; justify-content: center;
  color: #0a0a0a; font-size: 9px; font-weight: 700; margin-right: 3px;
}
.ena-ctx { color: var(--ena-ink2); }

/* ── Feeds ── */
.ena-feed { display: flex; flex-direction: column; gap: 4px; font-family: var(--ena-mono); }
.ena-evt {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  font-size: 12px; color: var(--ena-ink);
  padding: 6px 8px; border: 1px solid var(--ena-hairline); border-radius: 2px;
}
.ena-round-badge {
  font-size: 10px; font-weight: 700; color: var(--ena-muted);
  min-width: 32px; font-variant-numeric: tabular-nums;
}
.ena-trade-row {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  font-size: 11.5px; color: var(--ena-ink2);
  padding: 5px 8px; border-left: 2px solid var(--ena-muted);
}
.ena-trade-row .who { color: var(--ena-ink); font-weight: 600; }
.ena-outcome { font-weight: 700; font-size: 10.5px; letter-spacing: .05em; text-transform: uppercase; }
.ena-trade-accepted { border-left-color: var(--ena-good); }
.ena-trade-accepted .ena-outcome { color: var(--ena-good); }
.ena-trade-failed { border-left-color: var(--ena-serious); }
.ena-trade-failed .ena-outcome { color: var(--ena-serious); }
.ena-trade-rejected { border-left-color: var(--ena-critical); }
.ena-trade-rejected .ena-outcome { color: var(--ena-critical); }
.ena-trade-dropped { border-left-color: var(--ena-muted); }
.ena-trade-dropped .ena-outcome { color: var(--ena-muted); }

/* ── Benchmark tiles ── */
.ena-tiles { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.ena-tile {
  flex: 1 1 110px; background: var(--ena-page); border: 1px solid var(--ena-hairline);
  border-radius: 2px; padding: 10px 12px; min-width: 110px;
}
.ena-tile .v {
  font-size: 30px; color: var(--ena-ink);
  font-family: var(--ena-display); line-height: 1;
}
.ena-tile .v .unit { font-size: 13px; color: var(--ena-muted); margin-left: 2px; }
.ena-tile .l { font-size: 10px; color: var(--ena-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
.ena-prov { font-size: 11px; color: var(--ena-muted); }
.ena-prov.live { color: var(--ena-good); }

/* ── Hero ── */
.ena-hero { font-family: var(--ena-mono); padding: 4px 0 8px 0; }
.ena-hero .lcd {
  font-family: var(--ena-display); font-size: 44px; line-height: .95;
  color: var(--ena-ink); text-transform: uppercase; letter-spacing: .03em;
  margin: 0 0 8px 0;
}
.ena-hero .lcd .blink { color: var(--ena-critical); animation: ena-blink 1.4s steps(2) infinite; }
@keyframes ena-blink { 50% { opacity: 0; } }
.ena-hero .tag { font-size: 13px; color: var(--ena-ink2); margin: 0 0 12px 0; max-width: 860px; }
.ena-hero .stats { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; }
.ena-hero .stats .k { color: var(--ena-muted); }
.ena-hero .stats .v { color: var(--ena-ink); font-weight: 700; }
.ena-hero .notes { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.ena-hero .note { font-size: 11px; color: var(--ena-ink2); }

/* Backend health markdown */
.ena-health { font-size: 12px !important; }
.ena-health p, .ena-health li { color: var(--ena-ink2) !important; font-size: 12px !important; }
.ena-health h3 {
  font-family: var(--ena-display) !important; font-size: 20px !important;
  color: var(--ena-ink) !important; text-transform: uppercase;
}

/* ── Mobile responsiveness ── */
/* Horizontal-scroll shell for the vocabulary table; on desktop the table fits
   at 100% width so this is a no-op. */
.ena-table-wrap {
  width: 100%; max-width: 100%;
  overflow-x: auto; -webkit-overflow-scrolling: touch;
}
@media (max-width: 640px) {
  /* World map: ten flexible equal columns, square cells, card-width bound */
  .ena-grid {
    grid-template-columns: repeat(10, minmax(0, 1fr));
    grid-auto-rows: auto;
    width: 100%; max-width: 100%;
  }
  .ena-cell { aspect-ratio: 1; }
  /* Vocabulary table: keep columns readable, swipe inside .ena-table-wrap */
  .ena-table { min-width: 560px; }
  /* Panels: slightly tighter padding, headings/subtitles/axis wrap cleanly */
  .ena-card { padding: 10px 12px; }
  .ena-card-head { flex-wrap: wrap; gap: 4px 10px; }
  .ena-sub { white-space: normal; }
  .ena-axis { flex-wrap: wrap; gap: 2px 10px; }
  /* Gradio layout blocks must not force horizontal page overflow */
  .gradio-container .column, .gradio-container .row,
  .gradio-container .html-container, .gradio-container .prose {
    min-width: 0 !important; max-width: 100% !important;
  }
  .gradio-container { overflow-x: hidden; }
}
"""

DECO_ROW = ('<div class="ena-deco"><span>[</span>'
            '<span class="sq">■ ■ ■&nbsp;&nbsp;&nbsp;■ ■&nbsp;&nbsp;&nbsp;■ ■ ■'
            '&nbsp;&nbsp;&nbsp;■&nbsp;&nbsp;&nbsp;■</span>'
            '<span>]</span></div>')


def _esc(s) -> str:
    return html_lib.escape(str(s), quote=True)


def _res_var(resource: str) -> str:
    return f"var(--ena-res-{resource})"


def _agent_var(agent_id: str) -> str:
    return f"var(--ena-agent-{agent_id[-1]})"


def _agent_chip(agent_id: str, size: int = 15) -> str:
    return (f'<span class="ena-adopter" style="background:{_agent_var(agent_id)};'
            f'width:{size}px;height:{size}px" title="{_esc(agent_id)}">'
            f'{_esc(agent_id[-1])}</span>')


def _fmt_res(d: dict) -> str:
    if not isinstance(d, dict) or not d:
        return "?"
    return ", ".join(f"{v} {k}" for k, v in d.items())


def _card(head: str, body: str, sub: str = "", deco: bool = False) -> str:
    sub_html = f'<span class="ena-sub">{sub}</span>' if sub else ""
    deco_html = DECO_ROW if deco else ""
    return (f'<div class="ena-card">{deco_html}'
            f'<div class="ena-card-head">{head}{sub_html}</div>'
            f'{body}</div>')


# ─────────────────────────────────────────────
# UI read functions — all derive from the snapshot
# ─────────────────────────────────────────────

def _read_state():
    """One locked read of everything the dashboard needs."""
    with _sim_lock:
        sim = _sim
        running = _running
        error = _last_error
        backend = _active_backend
        reason = _backend_reason
    snapshot = sim.latest_snapshot if sim is not None else None
    return snapshot, running, error, backend, reason


def get_backend_status() -> str:
    now = time.time()
    if now - _probe_cache["t"] > _PROBE_TTL_S:
        _probe_cache["fireworks"] = check_fireworks_ready()
        _probe_cache["t"] = now

    fw_ok, fw_detail = _probe_cache["fireworks"]
    replay_ok = os.path.exists(REPLAY_FILE)
    _, _, error, backend, reason = _read_state()

    lines = [
        "### Backend health",
        f"- Fireworks (LLM API): {'🟢 ' + fw_detail if fw_ok else '🔴 ' + fw_detail}",
        f"- Replay sample: {'🟢 ' + REPLAY_FILE if replay_ok else '🔴 missing ' + REPLAY_FILE}",
        "- Heuristic: 🟢 always available (rule-based, no key)",
        f"- Active mode: **{backend or '—'}**" + (f" — {reason}" if reason else ""),
    ]
    if error:
        lines.append(f"- ⚠️ **Last error:** `{error}`")
    return "\n".join(lines)


def get_status() -> str:
    snapshot, running, error, _, _ = _read_state()
    if error:
        return f"🔴 Error — {error}"
    if snapshot is None:
        return "🟡 Starting… (waiting for first round)" if running else "⚪ Not started"
    if running:
        return (f"🟢 Running — Round {snapshot['round']} "
                f"| {snapshot['alive_count']}/3 agents alive "
                f"| backend: {snapshot['backend']}")
    return f"✅ Complete — {snapshot['round']} rounds | outputs saved to {OUTPUT_DIR}/"


def get_grid_display() -> str:
    """World map: 10×10 grid with zone tints, resource tiles, agent badges."""
    snapshot, _, _, _, _ = _read_state()
    if not snapshot:
        return _card("World map", '<div class="ena-empty">Simulation not started — '
                                  'press ▶ Start to see agents on the map.</div>', deco=True)

    tiles = {tuple(t["pos"]): t for t in snapshot.get("tiles", [])}
    # Agents can share a cell (e.g. both trading on a commons tile) — keep a
    # list per position so co-located agents render side by side, not lost.
    agents_at: dict = {}
    for a in snapshot["agents"]:
        if a.get("alive") and a.get("position"):
            agents_at.setdefault(tuple(a["position"]), []).append(a)

    def zone_of(r, c):
        return ("food" if c < 5 else "water") if r < 5 else ("metal" if c < 5 else "energy")

    cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            zone = zone_of(r, c)
            tip = [f"({r},{c}) · {zone} zone"]
            inner = ""
            tile = tiles.get((r, c))
            here = agents_at.get((r, c), [])
            if tile:
                if tile["amount"] > 0:
                    tip.append(f"{tile['type']} ×{tile['amount']}")
                else:
                    tip.append(f"{tile['type']} (depleted)")
            tip.extend(a["agent_id"] for a in here)

            if here:
                size = 24 if len(here) == 1 else 13
                inner += "".join(
                    f'<span class="ena-agent-badge" '
                    f'style="background:{_agent_var(a["agent_id"])};'
                    f'width:{size}px;height:{size}px;font-size:{12 if size > 20 else 9}px">'
                    f'{_esc(a["agent_id"][-1])}</span>'
                    for a in here
                )
                if tile and tile["amount"] > 0:
                    inner += (f'<span class="ena-dot ena-corner" '
                              f'style="background:{_res_var(tile["type"])}"></span>')
            elif tile:
                if tile["amount"] > 0:
                    size = 12 + min(6, tile["amount"])
                    inner += (f'<span class="ena-dot" style="background:{_res_var(tile["type"])};'
                              f'color:{RESOURCE_INK[tile["type"]]};'
                              f'width:{size}px;height:{size}px">'
                              f'{RESOURCE_LETTER[tile["type"]]}</span>')
                else:
                    inner += '<span class="ena-dot-empty"></span>'

            cells.append(
                f'<div class="ena-cell" title="{_esc(" · ".join(tip))}" '
                f'style="background:color-mix(in srgb, {_res_var(zone)} 7%, var(--ena-page))">'
                f'{inner}</div>'
            )

    legend = ['<div class="ena-legend">']
    for res in RESOURCES:
        legend.append(f'<span class="chip ena-bracket"><span class="ena-swatch" '
                      f'style="background:{_res_var(res)}"></span>{res}</span>')
    for a in snapshot["agents"]:
        state = "" if a["alive"] else " ✝"
        legend.append(f'<span class="chip ena-bracket"><span class="ena-swatch-sq" '
                      f'style="background:{_agent_var(a["agent_id"])}"></span>'
                      f'{_esc(a["agent_id"])}{state}</span>')
    legend.append('<span class="chip ena-bracket"><span class="ena-dot-empty"></span>depleted</span>')
    legend.append("</div>")

    body = f'<div class="ena-grid">{"".join(cells)}</div>{"".join(legend)}'
    return _card("World map",
                 body,
                 sub=f"round {snapshot['round']} · {snapshot['alive_count']}/3 alive · "
                     f"zone tint = home region of each resource",
                 deco=True)


def get_agent_cards() -> str:
    snapshot, _, _, _, _ = _read_state()
    if not snapshot:
        return _card("Agents", '<div class="ena-empty">Waiting for simulation…</div>')

    cards = ['<div class="ena-cards">']
    for agent in snapshot["agents"]:
        alive = agent["alive"]
        pill = ('<span class="ena-pill alive">● alive</span>' if alive
                else '<span class="ena-pill dead-pill">✝ dead</span>')
        rows = []
        for res in RESOURCES:
            val = agent["resources"].get(res, 0)
            pct = min(100, int(val / 20 * 100))
            low = ('<span class="ena-low">LOW</span>'
                   if alive and val <= 2 else "")
            rows.append(
                f'<div class="ena-bar-row"><span class="lbl">{res}{low}</span>'
                f'<div class="ena-track"><div class="ena-fill" '
                f'style="width:{pct}%;background:{_res_var(res)}"></div></div>'
                f'<span class="val">{val}</span></div>'
            )
        cards.append(
            f'<div class="ena-agent-card{"" if alive else " dead"}">'
            f'<div class="ena-agent-head">{_agent_chip(agent["agent_id"], 20)}'
            f'<span class="name">{_esc(agent["agent_id"])}</span>{pill}</div>'
            f'<div class="ena-score-row"><span class="ena-score">{agent["score"]:.1f}</span>'
            f'<span class="ena-score-lbl">score</span>'
            f'<span class="ena-trades-lbl">trades {agent["trades_completed"]}'
            f'/{agent["trades_attempted"]} done/proposed</span></div>'
            f'{"".join(rows)}</div>'
        )
    cards.append("</div>")
    return "".join(cards)


def get_activity_chart() -> str:
    """Per-round trade activity: bar height = proposals, split by outcome."""
    snapshot, _, _, _, _ = _read_state()
    history = (snapshot or {}).get("history") or []
    if not history:
        return _card("Activity", '<div class="ena-empty">No rounds recorded yet.</div>',
                     deco=True)

    max_att = max((h["attempted"] for h in history), default=1) or 1
    px_per = 100 // max_att  # chart is 120px tall, leave headroom
    bars = []
    for h in history:
        ok = h["succeeded"]
        miss = max(0, h["attempted"] - ok)
        tip = (f"round {h['round']} · {h['attempted']} proposed · "
               f"{ok} executed · {h['alive']}/3 alive")
        if h["attempted"] == 0:
            segs = '<span class="seg-none"></span>'
        else:
            segs = (f'<span class="seg-ok" style="height:{ok * px_per}px"></span>'
                    f'<span class="seg-miss" style="height:{miss * px_per}px"></span>')
        bars.append(f'<div class="bar" title="{_esc(tip)}">{segs}</div>')

    first_r, last_r = history[0]["round"], history[-1]["round"]
    axis = (f'<div class="ena-axis"><span>round {first_r}</span>'
            f'<span class="ena-bracket"><span class="ena-swatch-sq" '
            f'style="background:var(--ena-res-energy)"></span> executed</span>'
            f'<span class="ena-bracket"><span class="ena-swatch-sq" '
            f'style="background:var(--ena-res-food)"></span> not executed</span>'
            f'<span>round {last_r}</span></div>')
    return _card("Activity",
                 f'<div class="ena-chart">{"".join(bars)}</div>{axis}',
                 sub="trade proposals per round",
                 deco=True)


def get_vocabulary_table() -> str:
    snapshot, _, _, _, _ = _read_state()
    if not snapshot or not snapshot.get("vocab_entries"):
        return _card("Vocabulary", '<div class="ena-empty">No symbols invented yet — '
                                   'agents are still exploring.</div>')

    entries = sorted(snapshot["vocab_entries"],
                     key=lambda e: -e["stability_score"])[:12]
    rows = []
    for e in entries:
        adopters = "".join(_agent_chip(a) for a in e["adopters"])
        stab = e["stability_score"]
        meaning = e["dominant_context"] or e.get("inferred_meaning") or "—"
        rows.append(
            f'<tr><td><span class="ena-sym">{_esc(e["symbol"])}</span></td>'
            f'<td><div class="ena-stab"><div class="ena-track"><div class="ena-fill" '
            f'style="width:{int(stab * 100)}%;background:#e5e5e5"></div></div>'
            f'{stab:.2f}</div></td>'
            f'<td>{e["usage_count"]}</td>'
            f'<td>{e["success_rate"] * 100:.0f}%</td>'
            f'<td>{adopters}</td>'
            f'<td class="ena-ctx">{_esc(meaning)}</td></tr>'
        )
    table = ('<div class="ena-table-wrap"><table class="ena-table"><thead><tr>'
             '<th>Symbol</th><th>Stability</th><th>Uses</th><th>Success</th>'
             '<th>Adopters</th><th>Meaning (dominant context)</th>'
             '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")
    return _card("Vocabulary", table,
                 sub=f"{len(snapshot['vocab_entries'])} symbols invented · "
                     "top 12 by stability")


def get_convergence_events() -> str:
    snapshot, _, _, _, _ = _read_state()
    if not snapshot:
        return _card("Emergent words", '<div class="ena-empty">None yet.</div>')
    events = snapshot["convergence_events"]
    if not events:
        return _card("Emergent words",
                     '<div class="ena-empty">No convergence yet — a "word" is born when '
                     '2+ agents commit to the same symbol for the same trade context.</div>')
    rows = []
    for evt in reversed(events[-8:]):
        chips = "".join(_agent_chip(a) for a in evt["adopters"])
        rows.append(
            f'<div class="ena-evt"><span class="ena-round-badge">R{evt["round"]}</span>'
            f'<span class="ena-sym">{_esc(evt["symbol"])}</span>'
            f'<span class="ena-ctx">{_esc(evt["context"])}</span>{chips}</div>'
        )
    return _card("Emergent words", f'<div class="ena-feed">{"".join(rows)}</div>',
                 sub=f"{len(events)} convergence events · newest first")


def get_benchmark_display() -> str:
    snapshot, _, _, _, _ = _read_state()
    if not snapshot or not snapshot["benchmark"]:
        return _card("Benchmark", '<div class="ena-empty">No benchmark data yet.</div>')
    s = snapshot["benchmark"]
    live = snapshot["backend"] == "fireworks"
    prov = ('<div class="ena-prov live">● measured against a live LLM backend</div>' if live else
            f'<div class="ena-prov">⚠ {_esc(snapshot["backend"])} mode — timings are NOT '
            'live GPU measurements</div>')
    tiles = (
        f'<div class="ena-tiles">'
        f'<div class="ena-tile"><div class="v">{s["avg_parallel_inference_ms"]:.0f}'
        f'<span class="unit">ms</span></div><div class="l">avg parallel round '
        f'(3 agents at once)</div></div>'
        f'<div class="ena-tile"><div class="v">{s["parallel_speedup_factor"]}'
        f'<span class="unit">×</span></div><div class="l">vs sequential estimate '
        f'({s["avg_sequential_estimate_ms"]:.0f} ms)</div></div>'
        f'<div class="ena-tile"><div class="v">{s["total_trades_succeeded"]}'
        f'<span class="unit">/{s["total_trades_attempted"]}</span></div>'
        f'<div class="l">trades succeeded / attempted</div></div>'
        f'<div class="ena-tile"><div class="v">{s["total_symbols_invented"]}</div>'
        f'<div class="l">symbols invented</div></div>'
        f'</div>'
    )
    return _card("Benchmark", tiles + prov,
                 sub=f"{s['total_rounds']} rounds", deco=True)


def get_recent_trades() -> str:
    snapshot, _, _, _, _ = _read_state()
    if not snapshot or not snapshot["recent_trades"]:
        return _card("Trade log", '<div class="ena-empty">No trades yet.</div>')

    LABEL = {"accepted": "✓ accepted", "failed": "⚠ failed",
             "rejected": "✕ rejected", "dropped_invalid": "∅ dropped (invalid)",
             "dropped_collision": "∅ dropped (collision)"}
    rows = []
    for t in reversed(snapshot["recent_trades"]):
        ttype = t.get("type", "?")
        cls = ttype.split("_")[0] if ttype.startswith("dropped") else ttype
        detail = ""
        if ttype in ("accepted", "failed"):
            detail = f'gave {_esc(_fmt_res(t.get("offer")))} for {_esc(_fmt_res(t.get("request")))}'
        elif ttype == "rejected":
            detail = f'rejection symbol <span class="ena-sym">{_esc(t.get("rejection_symbol", "?"))}</span>'
        rows.append(
            f'<div class="ena-trade-row ena-trade-{cls}">'
            f'<span class="ena-round-badge">R{t["round"]}</span>'
            f'<span class="ena-outcome">{LABEL.get(ttype, _esc(ttype))}</span>'
            f'<span class="who">{_esc(t["proposer"][-1])} → {_esc(t["target"][-1])}</span>'
            f'{detail}'
            f'<span class="ena-sym">{_esc(t.get("symbol", "?"))}</span></div>'
        )
    return _card("Trade log", f'<div class="ena-feed">{"".join(rows)}</div>',
                 sub="last 12 events · newest first · digits are agent ids")


def refresh_all():
    return (
        get_status(),
        get_backend_status(),
        get_grid_display(),
        get_agent_cards(),
        get_activity_chart(),
        get_vocabulary_table(),
        get_convergence_events(),
        get_benchmark_display(),
        get_recent_trades(),
    )


def start_simulation(num_rounds: float, backend: str, seed: float, semantics: str) -> str:
    global _sim_thread, _sim, _running, _last_error, _active_backend, _backend_reason

    with _sim_lock:
        if _running:
            return "Simulation already running."

    # Preflight: never start a run that is guaranteed dead.
    try:
        resolved, reason = resolve_backend(backend, replay_file=REPLAY_FILE)
    except BackendUnavailable as e:
        with _sim_lock:
            _last_error = str(e)
        return f"❌ Cannot start: {e}"

    with _sim_lock:
        _running = True
        _last_error = None
        _sim = None
        _active_backend = resolved
        _backend_reason = reason
    _probe_cache["t"] = 0.0  # force a fresh health readout on next refresh

    _sim_thread = threading.Thread(
        target=_run_simulation_thread,
        args=(int(num_rounds), resolved, int(seed), semantics),
        daemon=True,
    )
    _sim_thread.start()
    return f"▶ Started on backend '{resolved}' — {reason}"


# ─────────────────────────────────────────────
# Gradio app
# ─────────────────────────────────────────────

HERO_HTML = """
<div class="ena-hero">
  <div class="lcd">Emergent Negotiation Arena <span class="blink">●</span></div>
  <p class="tag">Three AI agents in a resource-scarce grid world with no shared
  language. They must trade to survive — negotiating through symbols they invent,
  while a tracker measures which symbols stabilise into shared conventions
  ("words").</p>
  <div class="stats">
    <span class="ena-bracket"><span class="k">Agents:</span> <span class="v">3</span></span>
    <span class="ena-bracket"><span class="k">Grid:</span> <span class="v">10×10</span></span>
    <span class="ena-bracket"><span class="k">Resources:</span> <span class="v">4</span></span>
    <span class="ena-bracket"><span class="k">Backends:</span> <span class="v">5</span></span>
  </div>
  <div class="notes">
    <span class="note">■ <b>fireworks</b> = live LLM agents · no key? pick
    <b>replay</b> (recorded run) or <b>heuristic</b> (rule-based live run)</span>
    <span class="note">■ a "word" = a symbol committed to by 2+ agents for the
    same trade context</span>
    <span class="note">■ shared demo — everyone watching sees the same simulation</span>
  </div>
</div>
"""

# Client-side poll: gradio 4.44's `every=` re-runs silently fail on the load
# event ("Too many arguments provided for the endpoint"), so instead a small
# JS interval programmatically triggers the Refresh button's click event —
# the same server round-trip, on machinery that demonstrably works.
# Also force dark mode: this design is a single committed black-terminal look.
POLL_JS = """
() => {
  document.body.classList.add('dark');
  document.body.classList.remove('light');
  setInterval(() => {
    const btn = document.getElementById('ena-refresh-btn');
    if (btn) btn.click();
  }, 2500);
}
"""

THEME = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


def build_ui():
    with gr.Blocks(title="Emergent Negotiation Arena", theme=THEME,
                   css=CUSTOM_CSS, js=POLL_JS, head=FONT_HEAD) as demo:
        gr.HTML(HERO_HTML)

        with gr.Row():
            with gr.Column(scale=1):
                num_rounds = gr.Slider(10, 500, value=100, step=10, label="Number of rounds")
                backend = gr.Radio(
                    ["auto", "fireworks", "heuristic", "replay"],
                    value="auto",
                    label="Backend",
                    info="auto = best available | fireworks = LLM agents via cloud API | "
                         "heuristic = no-key rule-based demo | replay = recorded run",
                )
                semantics = gr.Radio(
                    ["visible", "hidden"],
                    value="visible",
                    label="Trade semantics shown to responder",
                    info="visible = demo mode (responder sees offer/request) | "
                         "hidden = research mode (responder sees only the symbol + hints)",
                )
                seed = gr.Number(value=42, label="Random seed")
                start_btn = gr.Button("▶ Start Simulation", variant="primary")
                start_msg = gr.Textbox(label="Start result", interactive=False)
                refresh_btn = gr.Button("🔄 Refresh Dashboard", elem_id="ena-refresh-btn")
                backend_status = gr.Markdown(get_backend_status(), elem_classes=["ena-health"])

            with gr.Column(scale=2):
                status_display = gr.Textbox(label="Simulation status", interactive=False)
                grid_display = gr.HTML(get_grid_display())

        agent_display = gr.HTML(get_agent_cards())
        activity_display = gr.HTML(get_activity_chart())

        with gr.Row():
            with gr.Column(scale=3):
                vocab_display = gr.HTML(get_vocabulary_table())
            with gr.Column(scale=2):
                convergence_display = gr.HTML(get_convergence_events())

        with gr.Row():
            with gr.Column():
                benchmark_display = gr.HTML(get_benchmark_display())
            with gr.Column():
                trades_display = gr.HTML(get_recent_trades())

        all_outputs = [
            status_display,
            backend_status,
            grid_display,
            agent_display,
            activity_display,
            vocab_display,
            convergence_display,
            benchmark_display,
            trades_display,
        ]

        # Events
        start_btn.click(
            fn=start_simulation,
            inputs=[num_rounds, backend, seed, semantics],
            outputs=[start_msg],
        )

        # show_progress="hidden": the poll re-runs this every ~2.5s — default
        # progress UI would flash a skeleton over every panel on each tick.
        refresh_btn.click(fn=refresh_all, inputs=[], outputs=all_outputs,
                          show_progress="hidden")

        # Populate the dashboard once on page load; the POLL_JS interval
        # keeps it fresh afterwards by clicking the refresh button.
        demo.load(fn=refresh_all, inputs=[], outputs=all_outputs,
                  show_progress="hidden")

    return demo


# Module-level Blocks object: Hugging Face Spaces (sdk: gradio, app_file:
# ui/app.py) imports this file and serves `demo` — it never runs __main__.
demo = build_ui()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
