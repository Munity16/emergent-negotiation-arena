# Hackathon Submission Material

Ready-to-paste content for the submission form.
Repository: https://github.com/Munity16/emergent-negotiation-arena

## Project title

Emergent Negotiation Arena

## Short description (one sentence)

Three LLM agents trapped in a scarce grid-world economy with no shared
language invent trade symbols from scratch — and a live dashboard flags the
exact round a shared "word" is born.

## Long description

Emergent Negotiation Arena is a survival economy for three LLM agents. Each
agent starts rich in one resource and poor in the rest, on a 10×10 map whose
resources spawn in zones — nobody can survive alone. Agents trade to live,
and every trade proposal carries a symbol the agent invents itself: any
string, never explained. Agents learn what symbols mean only from which
trades succeed.

A vocabulary tracker measures every symbol's usage, success rate, stability,
and adopters. When two or more agents commit to the same symbol for the same
trade context (by proposing with it or accepting under it — merely receiving
it doesn't count), the system fires a convergence event: a word has emerged.
In a logged live run on the Fireworks API (gpt-oss-120b, 30 rounds, seed 7),
agents invented `Z1` for water→metal trades — both parties adopted it, with
the convergence event at round 8 — and one agent coined `W2F` for
water→food. 4 of 4 proposed trades executed, with 0 fallback decisions.

Everything is engineered to be auditable. The world — not the model — is the
referee: an "accepted" trade the buyer can't pay for is recorded as failed
everywhere, and malformed LLM output is validated and dropped visibly. Runs
are seeded, fully logged, and a replay backend re-feeds a recorded log
through the real simulation byte-exactly, so every claim is reproducible.
Two demo modes need no key at all (deterministic heuristic agents and
recorded replay), so the app always works; a hidden-semantics research mode
withholds trade details so symbols must genuinely carry meaning. A
terminal-styled Gradio dashboard shows the map, agents, per-round trade
activity, the growing vocabulary, convergence events, and measured
parallel-inference latency, live.

The dashboard is deployed as a Render Web Service (free instance,
Python 3.12.8) with auto-deployment from the repository's `main` branch.
Verified on the deployed service: page load, the Replay backend, the
Heuristic backend with its background simulation thread, and access from a
separate mobile device. The Fireworks backend is configured for secure
runtime use through Render's environment settings (the key is never
committed); its final live deployment test is pending. Free instances sleep
after inactivity, so the first load may incur a cold-start delay.

## Technology tags

`multi-agent` · `emergent-communication` · `LLM` · `Fireworks AI` ·
`gpt-oss-120b` · `Python` · `asyncio` · `Gradio` · `Docker` · `simulation` ·
`parallel-inference` · `pytest`

## Key innovation

Auditable emergence: symbol conventions arise from survival pressure alone
(no scripted vocabulary), and every emergence claim is backed by seeded,
byte-exact-replayable logs. Adoption is defined by *commitment* (proposing
or accepting under a symbol), not mere exposure — so a convergence event is
evidence of a working convention, not a coincidence counter.

## Business / practical value

- A testbed for studying how autonomous agents develop coordination
  protocols under resource pressure — relevant to multi-agent systems that
  must negotiate (marketplaces, logistics, resource allocation).
- A reference pattern for production multi-agent LLM serving: parallel
  decision rounds via `asyncio.gather`, strict validation of untrusted model
  output, truthful outcome logging, and deterministic replay for debugging
  and audits.
- A teaching artifact: emergent communication is watchable in minutes, with
  honest labelling of what is demo versus research-grade evidence.

## How the AI technology is used

Three independent LLM agents (Fireworks AI API, default `gpt-oss-120b`,
`reasoning_effort=low`) each receive a private observation and a private
symbol memory, and return a structured JSON decision — move, collect,
propose a trade with an invented symbol, or respond to one. All three calls
are issued in parallel each round; one round costs one API round-trip
instead of three. Output is parsed defensively (reasoning-block stripping,
JSON extraction, strict resource validation) with safe fallbacks, and
per-round parallel-vs-sequential latency is measured and displayed. Backends
are preflighted with automatic fallback (fireworks → replay → heuristic) so
the demo cannot dead-end.

## Demo instructions

```bash
pip install -r requirements.txt

# Guaranteed demo, no key: replays a recorded 60-round run
python main.py --backend replay

# Replay the real LLM run (invented symbols Z1, W2F)
python main.py --backend replay --replay-file outputs/sample_fireworks_run.json

# Live LLM agents (needs a Fireworks key)
export FIREWORKS_API_KEY=your_key
python main.py --backend fireworks --rounds 50

# Dashboard: pick a backend, press Start, watch the vocabulary form
python main.py --ui        # → http://localhost:7860
```

What to watch for in the dashboard: the world map (agents converge on the
central commons), the trade log turning green, and the *Emergent words*
panel — each entry is a convention two agents committed to.

## Test status

84/84 automated tests passing (world mechanics, hostile-input validation,
outcome truthfulness, adoption/convergence semantics, determinism,
byte-exact replay, backend resolution).
