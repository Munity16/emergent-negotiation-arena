"""
main.py — Emergent Negotiation Arena

CLI entrypoint. Run the simulation and save outputs.

Usage:
    # Auto-pick the best available backend (fireworks → replay → heuristic)
    python main.py --rounds 100

    # Run 50 rounds on the Fireworks AI API (LLM agents)
    python main.py --rounds 50 --backend fireworks

    # No key: deterministic rule-based demo
    python main.py --rounds 50 --backend heuristic

    # Guaranteed demo: replay a recorded run exactly
    python main.py --backend replay --replay-file outputs/sample_run.json

    # Launch the Gradio UI instead
    python main.py --ui

Environment variables:
    FIREWORKS_API_KEY   API key for Fireworks AI
    FIREWORKS_MODEL     Fireworks model id (default: accounts/fireworks/models/gpt-oss-120b)
    AGENT_BACKEND       Backend if --backend not given (overridden by --backend flag)
    SEMANTICS_MODE      "visible" (demo) or "hidden" (research) — see --semantics
"""

import argparse
import asyncio
import os
import sys

# Windows consoles often default to a legacy code page (cp1252) that cannot
# encode the box-drawing/arrow characters in status output — reconfigure
# instead of crashing mid-simulation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REPLAY_FILE = "outputs/sample_run.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Emergent Negotiation Arena"
    )
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of simulation rounds (default: 100; ignored for replay)")
    parser.add_argument("--backend",
                        choices=["auto", "fireworks", "heuristic", "replay"],
                        default="auto",
                        help="LLM backend: 'auto' picks the best available "
                             "(fireworks → replay → heuristic); "
                             "'fireworks' = LLM agents via cloud API; "
                             "'heuristic' = rule-based no-key demo; "
                             "'replay' = play back a recorded run")
    parser.add_argument("--replay-file", type=str, default=DEFAULT_REPLAY_FILE,
                        help=f"Recorded run to replay (default: {DEFAULT_REPLAY_FILE})")
    parser.add_argument("--replay-delay", type=float, default=0.05,
                        help="Pause per replayed round in seconds (default: 0.05)")
    parser.add_argument("--semantics", choices=["visible", "hidden"], default="visible",
                        help="'visible': responder sees the full offer/request (demo mode). "
                             "'hidden': responder sees only the symbol + coarse hints and must "
                             "infer meaning from experience (research mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save output files")
    parser.add_argument("--ui", action="store_true",
                        help="Launch Gradio UI instead of CLI simulation")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print status every 10 rounds")
    return parser.parse_args()


async def run_cli(args):
    from agents.llm_agent import resolve_backend, BackendUnavailable
    from core.simulation import Simulation

    os.environ["SEMANTICS_MODE"] = args.semantics

    try:
        backend, reason = resolve_backend(args.backend, replay_file=args.replay_file)
    except BackendUnavailable as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    os.environ["AGENT_BACKEND"] = backend
    print(f"\nBackend: {backend.upper()} — {reason}")
    print(f"Semantics mode: {args.semantics}")

    agent_pool = None
    num_rounds, seed = args.rounds, args.seed
    output_dir = args.output_dir

    if backend == "replay":
        from core.replay import ScriptedAgentPool, load_replay_file, ReplayFormatError
        try:
            meta, rounds = load_replay_file(args.replay_file)
        except ReplayFormatError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        agent_pool = ScriptedAgentPool(rounds, playback_delay_s=args.replay_delay)
        num_rounds = len(rounds)
        seed = meta.get("seed", args.seed)
        # Keep replay artefacts out of the way of the original recording.
        output_dir = os.path.join(args.output_dir, "replay")
        print(f"Replaying {num_rounds} recorded rounds "
              f"(original backend: {meta.get('backend', 'unknown')}, seed: {seed})")
    elif backend == "fireworks":
        print(f"Fireworks model: {os.getenv('FIREWORKS_MODEL', 'accounts/fireworks/models/gpt-oss-120b')}")

    sim = Simulation(
        num_rounds=num_rounds,
        seed=seed,
        verbose=args.verbose,
        output_dir=output_dir,
        agent_pool=agent_pool,
    )

    try:
        await sim.run()
    finally:
        await sim.agent_pool.close()
        sim.save_outputs()


def run_ui():
    sys.path.insert(0, ".")
    from ui.app import demo  # built once at module level (shared with HF Spaces)
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    args = parse_args()
    if args.ui:
        run_ui()
    else:
        asyncio.run(run_cli(args))
