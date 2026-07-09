---
title: Emergent Negotiation Arena
emoji: 🧠
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: ui/app.py
pinned: true
license: mit
---

# 🧠 Emergent Negotiation Arena

> Three agents are dropped into a resource-scarce 10×10 grid world.
> They must trade to survive, and every trade proposal carries a **symbol the
> agent invents itself**. The tracker watches which symbols get reused,
> adopted by other agents, and stabilised into shared conventions —
> and flags the moment a "word" is born.

**What this demonstrates, stated precisely:** agents developing *symbolic
negotiation conventions* — reused, multi-agent-adopted tokens tied to specific
trade contexts. In the default demo mode the responder also sees the concrete
offer, so symbols are labels for deals rather than a load-bearing language. A
stricter **hidden-semantics research mode** (`--semantics hidden`) withholds
the offer details, so the responder must act on the symbol plus its own past
experience — there, symbol meaning actually carries information.

---

## Quick start

Works on any machine with Python 3.11+:

```bash
pip install -r requirements.txt

# 1. LLM agents (the full experience — needs a Fireworks AI key)
export FIREWORKS_API_KEY=your_key      # never commit this
python main.py --backend fireworks --rounds 50

# 2. No key? Watch a guaranteed demo: replays a recorded 60-round run exactly
python main.py --backend replay

# 3. Or run a live simulation with the deterministic rule-based backend
python main.py --backend heuristic --rounds 60 --seed 42

# 4. Or launch the Gradio dashboard (pick backend in the UI)
python main.py --ui        # → http://localhost:7860
```

`--backend auto` (the default) probes what's available and picks the best:
**Fireworks → replay → heuristic**, so the same command works with or
without an API key.

## Backends

| Backend | Needs | What it is |
|---|---|---|
| `fireworks` | `FIREWORKS_API_KEY` | 3 LLM agents (gpt-oss-120b) via the Fireworks AI API, decided in parallel |
| `heuristic` | nothing | Deterministic rule-based agents; exercises the entire trade/symbol pipeline |
| `replay` | nothing | Re-feeds a recorded run's decisions through the real simulation — byte-exact reproduction |
| `auto` | nothing | Preflights the above in order and picks the first healthy one |

Every backend is preflighted before the simulation starts: an explicitly
requested backend that isn't available fails fast with a clear error instead
of dying mid-run; `auto` falls through to the next option and tells you why.

---

## How it works

- **World:** 10×10 grid, four resources (food, water, metal, energy) spawned
  in quadrant zones plus a small central commons. Agents consume resources
  every few rounds and die if starved.
- **Asymmetry forces trade:** each agent starts rich in one specialty and
  poor elsewhere, and resource tiles are zoned — nobody can survive
  comfortably alone.
- **Rounds:** every round all agents decide **in parallel**
  (`asyncio.gather`) — move, collect, propose a trade (with an invented
  symbol), or respond to one. The world executes trades only if both sides
  can actually pay; the *executed* outcome, not the agent's intent, is what
  reaches the vocabulary tracker and both agents' memories.
- **Vocabulary tracking:** a symbol's *adopters* are agents that committed to
  it — proposed with it, or accepted a trade under it. Merely receiving a
  proposal does not count. When 2+ agents commit to the same symbol for the
  same resource context, that's a **convergence event**. Symbols unused for
  10 rounds are marked extinct.

### Semantics modes

| Mode | Responder sees | Honest claim |
|---|---|---|
| `visible` (default) | symbol **and** full offer/request | symbolic conventions form over deals; great for demos |
| `hidden` | symbol + offer-size hint + own past experience with that symbol | symbols must carry meaning; this is the research setting |

---

## Sample results (real run, included in the repo)

`outputs/sample_run.json` / `sample_vocabulary.json` / `sample_benchmark.json`
come from an actual run: **heuristic backend, seed 42, 60 rounds** — they are
regenerable with the quick-start command above, and `--backend replay`
reproduces them exactly.

- 19 trades proposed, 9 executed; 8 symbols invented, 4 convergence events,
  8 extinctions
- Final scores: agent_0 = 99.0 (alive), agent_1 = 117.0 (alive),
  agent_2 = 29.0 (starved — death is real in this economy)

```
Symbol   Uses  Success%  Stability  Adopters                    Context
EW2         9     44.4%      0.781  agent_0, agent_1, agent_2   energy→water
NAK        10      0.0%      0.667  agent_0, agent_1, agent_2   (rejection marker)
WE2         4     75.0%      0.606  agent_0, agent_2            water→energy
WE1         1    100.0%      0.589  agent_0, agent_1            water→energy
FE2         1    100.0%      0.589  agent_0, agent_2            food→energy
```

Convergence events: `WE1` (round 10), `EW2` (round 12), `FE2` (round 14),
`WE2` (round 15).

**Provenance caveats, so nobody is misled:**

- The heuristic backend mints symbols from a fixed template
  (offer-initial + request-initial + agent number, e.g. `EW2`), so this run
  demonstrates the *pipeline* — adoption, convergence, extinction mechanics —
  not LLM-emergent naming. Novel symbol invention requires the `fireworks`
  backend.
- The timing fields in `sample_benchmark.json` (~0.2 ms/round) measure
  rule-based decisions, **not LLM inference**. The "sequential estimate" is
  parallel-time × 3 *by construction*; the parallelism claim is only
  meaningful on a real LLM backend. The dashboard labels non-LLM timings
  accordingly.

---

## Fireworks AI (LLM agents)

```bash
export FIREWORKS_API_KEY=your_key          # never commit this
export FIREWORKS_MODEL=accounts/fireworks/models/gpt-oss-120b   # optional override
python main.py --rounds 50 --backend fireworks
```

All three agents decide **in parallel** (`asyncio.gather`), so a round costs
one API round-trip instead of three. Requests use the Fireworks
OpenAI-compatible endpoint with retries and exponential backoff on 429/5xx.
If the key is missing, `--backend fireworks` fails fast at startup; `auto`
simply skips it.

## Docker

```bash
cd docker

# LLM agents:
FIREWORKS_API_KEY=your_key docker compose up

# No key: auto-falls back to replay/heuristic (UI on :7860)
docker compose up
```

The image bundles the sample run, so the replay demo works inside a
fresh container with no key.

---

## Project structure

```
emergent-negotiation-arena/
├── core/
│   ├── grid_world.py      # 10×10 environment, zones+commons, trade execution, validation
│   ├── symbol_tracker.py  # Vocabulary registry, adoption/convergence/extinction
│   ├── simulation.py      # Round orchestration, truthful trade logging, UI snapshots
│   ├── replay.py          # ScriptedAgentPool: byte-exact replay of recorded runs
│   └── benchmark.py       # Standalone parallel-vs-sequential benchmark script
├── agents/
│   └── llm_agent.py       # Fireworks LLM client, heuristic policy, backend resolver
├── ui/
│   ├── app.py             # Gradio dashboard: backend health panel, grid, vocab, trades
│   ├── vocabulary_viz.py  # D3.js vocabulary lineage tree (HTML export)
│   └── benchmark_card.py  # Shareable PNG benchmark card
├── tests/                 # world, tracker, validation, replay, backend tests
├── scripts/
│   └── demo_video_run.py  # 30-round highlight reel for screen recording
├── docker/                # Dockerfile (healthcheck), compose, .dockerignore
├── main.py                # CLI entrypoint
└── outputs/
    ├── sample_run.json          # recorded 60-round run (powers --backend replay)
    ├── sample_vocabulary.json   # its vocabulary/convergence data
    └── sample_benchmark.json    # its timing record (heuristic — not GPU numbers)
```

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Coverage includes: malformed/hostile trade payloads (negative amounts,
unknown resources, wrong types), trade-outcome truthfulness (an "accepted"
trade that can't be paid is recorded as failed everywhere), adoption and
convergence semantics, heuristic-run determinism (two same-seed runs produce
identical logs), replay fidelity (replay reproduces a recorded run exactly),
and backend resolution/fail-fast behaviour with no network and no key.

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `AGENT_BACKEND` | `auto` | Backend if `--backend` not given |
| `FIREWORKS_API_KEY` | — | Fireworks credential (env only — never hardcoded) |
| `FIREWORKS_MODEL` | `accounts/fireworks/models/gpt-oss-120b` | Fireworks model id |
| `SEMANTICS_MODE` | `visible` | `visible` demo / `hidden` research |
| `REPLAY_FILE` | `outputs/sample_run.json` | Recording used by the UI's replay mode |
| `OUTPUT_DIR` | `outputs` | Where run artefacts are written |
| `PORT` | `7860` | Gradio port |

CLI flags mirror these: `--rounds`, `--backend`, `--seed`, `--semantics`,
`--replay-file`, `--replay-delay`, `--output-dir`, `--ui`.

## Outputs

Each run writes to `outputs/`:

- **simulation_log.json** — meta (backend, seed, semantics) + full per-round
  log: decisions, trades (accepted / failed / rejected / dropped-invalid /
  dropped-collision), world state, vocabulary summary
- **vocabulary.json** — every symbol with usage, success rate, stability,
  adopters, dominant context; convergence + extinction events
- **benchmark.json** — per-round inference latency and trade counts

## Known limitations

- In `visible` mode the emergent-language claim is deliberately modest
  (conventions, not language); use `--semantics hidden` for the stronger
  setting.
- Heuristic symbols are template-minted (see provenance caveats above).
- Agents can and do die: the economy was tuned (collection rates, respawn,
  consumption interval, central commons) so that a 60-round no-key run is
  watchable, but survival is not guaranteed — that pressure is what makes
  trade worth inventing words for.
- Benchmark "speedup" compares measured parallel latency against a
  sequential *estimate* (×3), not a measured sequential run.

## License

MIT.
