<p align="center">
  <img src="assets/branding/logo.svg" width="170" alt="Emergent Negotiation Arena logo">
</p>

<p align="center">
  <img src="assets/branding/hero-banner.svg" width="100%" alt="Emergent Negotiation Arena — three AI agents, no shared language, they must invent one to survive">
</p>

<p align="center"><b>Three LLM agents, trapped in a scarce world with no shared language, invent one — live, measurably, on screen.</b></p>

<p align="center">
  <img src="assets/badges/tests.svg" height="26" alt="tests 84/84">
  <img src="assets/badges/python.svg" height="26" alt="python 3.11+">
  <img src="assets/badges/gradio.svg" height="26" alt="gradio 4.x">
  <img src="assets/badges/fireworks.svg" height="26" alt="llm fireworks ai">
  <img src="assets/badges/license.svg" height="26" alt="license MIT">
</p>

<p align="center">
  <img src="assets/demo/arena-loop.gif" width="760" alt="Animated loop: three agents exchange invented symbols until a repeated symbol stabilises into a shared word">
</p>

## Overview

Three autonomous agents begin in a resource-scarce 10×10 world with **no
shared language**. To survive they must trade — and every trade proposal
carries a symbol the agent invents itself: any string, never explained.
The system tracks every symbol's usage, success rate, and adopters, and
flags the exact round a repeated symbol stabilises into a shared
convention: a primitive "word".

```
NO SHARED LANGUAGE
  → SURVIVAL PRESSURE
    → TRADE
      → INVENTED SYMBOLS
        → REPEATED USE
          → SHARED CONVENTION
            → PRIMITIVE "WORD"
```

## The Research Question

Multi-agent AI systems need to coordinate, but handing agents a pre-built
protocol hides the interesting question: *can communication conventions
emerge from pressure alone?* Most emergent-language work is either a toy
gridworld with scripted signals or a paper without a running artifact.
This project is something you can watch — and audit.

## How the Experiment Works

A survival economy where communication has to pay for itself:

- **10×10 grid world**, four resources spawned in zones plus a central commons
- Each agent starts rich in one resource, poor in the rest — nobody survives alone
- Every trade proposal carries an **invented symbol** (any string, never explained)
- Agents learn what symbols mean only from which trades succeed
- Starvation is real: agents that fail to trade die

```
every round
  ├─ all 3 agents decide IN PARALLEL (one asyncio.gather = one API round-trip)
  │    move · collect · propose_trade(offer, request, SYMBOL) · accept/reject
  ├─ the world executes trades — only if both sides can actually pay
  ├─ the vocabulary tracker records usage, success, adopters per symbol
  └─ 2+ agents committed to the same symbol, same context → CONVERGENCE EVENT
```

## How a Shared Word Emerges

- **Adoption means commitment**: proposing a trade with a symbol, or
  accepting a trade under it. Merely observing or receiving a symbol never
  counts as adoption.
- **Convergence** fires when 2+ committed agents use the same symbol for the
  same trade context — evidence of a working convention, not a coincidence
  counter.
- **Extinction**: symbols unused for 10 rounds are marked extinct, so the
  vocabulary reflects living conventions, not accumulated noise.

Two semantics modes keep the claims honest: default `visible` mode is a demo
(the responder sees the offer; symbols are conventions over deals).
`--semantics hidden` is the research setting: the responder sees *only the
symbol* plus its own history with it, so meaning must genuinely be carried
by the token.

## Architecture

```
main.py                  CLI entrypoint (run / replay / dashboard)
core/grid_world.py       world: zones, tiles, trade execution, strict validation
core/simulation.py       round loop, truthful logging, thread-safe UI snapshots
core/symbol_tracker.py   vocabulary registry: adoption, convergence, extinction
core/replay.py           ScriptedAgentPool — byte-exact replay of recorded runs
agents/llm_agent.py      Fireworks client, prompts, parsing, heuristic policy,
                         backend resolver (preflight + fallback)
ui/app.py                terminal-styled Gradio dashboard
tests/                   84 automated tests
```

The simulation runs in a **background thread**; the dashboard reads only
**immutable per-round snapshots**, swapped atomically — the UI can never
observe a half-updated round.

## The World Is the Referee

LLM output is treated as **untrusted input**. The world — not the model —
decides what happened:

- An agent "accepting" a trade it cannot pay for is recorded as a *failed*
  trade everywhere.
- Unknown resources, negative amounts, and proposal collisions are validated
  and dropped visibly.
- Every decision, trade, and adoption event is logged; the replay backend
  re-feeds a recorded log through the *real* simulation and reproduces it
  byte-exactly. Claims are checkable, not vibes.

## Execution Backends

The demo can never dead-end:

| Backend | Needs | What it is |
|---|---|---|
| `fireworks` | `FIREWORKS_API_KEY` | live LLM agents (`gpt-oss-120b`), parallel decisions |
| `heuristic` | nothing | deterministic rule-based agents — full pipeline, no key |
| `replay` | nothing | byte-exact re-run of a recorded log through the real simulation |
| `auto` | nothing | preflights the above in order, picks the first healthy one |

## Verified Results

From a real, logged Fireworks run (`gpt-oss-120b`, 30 rounds, seed 7 —
included at [`outputs/sample_fireworks_run.json`](outputs/sample_fireworks_run.json)):

- Agents invented **`Z1`** for water→metal trades — both parties adopted it,
  with the convergence event at **round 8**
- One agent coined **`W2F`** for water→food
- **4 of 4** proposed trades executed; **0** fallback decisions across the run

From the bundled 60-round heuristic run: **9 executed trades, 8 symbols,
4 convergence events** — reproduced byte-exactly by `--backend replay`.

Provenance is always labelled: non-LLM timings in the dashboard are marked
"NOT live measurements", and the heuristic's template-minted symbols are
never presented as LLM-emergent naming.

## Live Deployment Status

The dashboard is deployed as a **Render Web Service** (free instance,
Python 3.12.8, started with `python ui/app.py`), with auto-deployment from
this repository's `main` branch. Verified on the deployed service so far:

- ✅ public page loads with correct branding and encoding
- ✅ Replay backend runs end-to-end
- ✅ Heuristic backend runs live rounds with the background simulation thread
- ✅ opened successfully from a separate mobile device

The Fireworks backend on the deployed service is configured for secure
runtime use (the key is supplied through Render's environment settings, never
committed); its final live deployment test is pending.

Note: free Render instances sleep after inactivity, so the first request
after a quiet period incurs a cold-start delay. The public URL will be added
here after final mobile verification.

## Running Locally

```bash
git clone https://github.com/Munity16/emergent-negotiation-arena.git
cd emergent-negotiation-arena
pip install -r requirements.txt

# Guaranteed demo, no key — replays a recorded 60-round run
python main.py --backend replay

# The dashboard: pick a backend, press Start, watch the vocabulary form
python main.py --ui        # → http://localhost:7860
```

Works on Windows, macOS, and Linux (Python 3.11+). Docker alternative:

```bash
cd docker
FIREWORKS_API_KEY=your_key docker compose up   # LLM agents
docker compose up                              # no key → replay/heuristic
```

No Python at all? [`web/index.html`](web/index.html) is a zero-backend
**static replay viewer** that plays the recorded runs (including the real
LLM run) in the browser.

## Fireworks Configuration

Live LLM agents need a Fireworks AI key, read from the environment only:

```bash
cp .env.example .env       # fill in FIREWORKS_API_KEY (or export it)
python main.py --backend fireworks --rounds 50
```

On Windows PowerShell: `$env:FIREWORKS_API_KEY = "your_key"`.

| Env var | Default | Meaning |
|---|---|---|
| `FIREWORKS_API_KEY` | — | Fireworks credential (env only — never committed) |
| `FIREWORKS_MODEL` | `accounts/fireworks/models/gpt-oss-120b` | model override |
| `AGENT_BACKEND` | `auto` | backend when `--backend` not given |
| `SEMANTICS_MODE` | `visible` | `visible` demo / `hidden` research |
| `REPLAY_FILE` | `outputs/sample_run.json` | recording used by replay |
| `PORT` | `7860` | dashboard port |

All three agent decisions are issued concurrently, so a round costs one API
round-trip instead of three. Requests retry with exponential backoff on
429/5xx; a missing key fails fast at startup instead of dying mid-run.

## Replay and Deterministic Experiments

Runs are seeded and fully logged; the replay backend re-feeds a recorded log
through the real simulation, byte-exactly:

```bash
# Replay the bundled heuristic run
python main.py --backend replay

# Replay the real LLM run (watch Z1 and W2F emerge)
python main.py --backend replay --replay-file outputs/sample_fireworks_run.json
```

In the dashboard, watch the trade log turn green and the *Emergent words*
panel — each entry is a convention two agents committed to.

## Testing

```bash
pytest tests/ -q           # 84 passed
```

**84/84 automated tests passing**: world mechanics, hostile-input
validation, outcome truthfulness, adoption/convergence semantics,
determinism, byte-exact replay, and backend resolution.

## Technology Stack

Python 3.11+ (3.12.8 on Render) · asyncio · httpx · Gradio 4 ·
Fireworks AI (`gpt-oss-120b`) · matplotlib · Docker · pytest

## Repository Structure

```
main.py                  CLI entrypoint
core/                    grid_world · simulation · symbol_tracker · replay · benchmark
agents/                  llm_agent.py — Fireworks client, heuristic policy, resolver
ui/                      app.py dashboard · vocabulary_viz · benchmark_card
tests/                   84 automated tests
outputs/                 curated sample runs (incl. the logged Fireworks run)
web/index.html           zero-backend static replay viewer
docker/                  Dockerfile + docker-compose
render.yaml              Render Web Service blueprint
deploy/huggingface/      alternative Hugging Face Space configuration
scripts/                 demo/video helper scripts
assets/                  branding, badges, demo gif
```

## AMD Hackathon Relevance

The core workload is **parallel multi-agent LLM inference**: every round is
three agent decisions executed concurrently with `asyncio.gather`, with
measured parallel-vs-sequential latency reported per round in the dashboard
and the benchmark card. That is precisely the serving pattern large-memory
accelerators are built for. The LLM client speaks the OpenAI-compatible chat
API — currently pointed at Fireworks serverless — and the backend resolver
makes swapping the serving endpoint a configuration change, not a rewrite.

## Security

- `FIREWORKS_API_KEY` is read from environment variables only — never
  hard-coded, logged, or committed.
- `.gitignore` blocks `.env` and local launchers; no secrets are stored in
  GitHub.
- [`.env.example`](.env.example) documents variable names only, without values.
- On Render, the key lives in the service's secure environment settings.

## Limitations and Honest Scope

- This is a research demo, not a production system: 3 agents, a 10×10 world,
  short runs.
- `W2F` was coined and used, but is **not** claimed as converged — only `Z1`
  produced a verified convergence event in the logged Fireworks run.
- The default `visible` semantics mode is a demo setting; `hidden` mode is
  the stricter research configuration.
- Heuristic-backend symbols are template-minted, not LLM-emergent, and are
  labelled as such; non-LLM timings are marked "NOT live measurements".
- The application does not currently run on AMD hardware; the relevance is
  the honestly-described parallel-inference serving pattern above.
- Free-tier hosting sleeps after inactivity (cold-start delay on first load).

## License

MIT — see [LICENSE](LICENSE). Built as an AMD hackathon submission.
