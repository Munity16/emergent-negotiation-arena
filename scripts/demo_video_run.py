"""
scripts/demo_video_run.py — Emergent Negotiation Arena

A 30-round "highlight reel" run designed for screen-recording the demo video.

For each round it prints:
  - the ASCII grid + agent resource bars
  - the vocabulary table so far
  - any convergence events that happened THIS round, highlighted in green

...then pauses ~0.5s so a screen recording reads naturally, before ending
with the final vocabulary table and a benchmark card.

Usage:
    python scripts/demo_video_run.py
    python scripts/demo_video_run.py --rounds 30 --pause 0.5 --backend fireworks
    python scripts/demo_video_run.py --backend heuristic   # no API key needed
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows consoles often default to cp1252, which cannot encode the
# box-drawing/arrow characters this script prints — reconfigure, don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GREEN = "\033[1;32m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"


def _banner(text: str):
    width = 72
    print(f"{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}{text.center(width)}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}\n")


async def run_demo(num_rounds: int, pause_s: float, backend: str, seed: int,
                   output_dir: str, clear: bool):
    from agents.llm_agent import resolve_backend, BackendUnavailable
    from core.simulation import Simulation

    try:
        backend, reason = resolve_backend(backend)
    except BackendUnavailable as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    os.environ["AGENT_BACKEND"] = backend
    print(f"Backend: {backend} — {reason}")

    sim = Simulation(num_rounds=num_rounds, seed=seed, verbose=False, output_dir=output_dir)

    if clear:
        print(CLEAR_SCREEN, end="")
    _banner("EMERGENT NEGOTIATION ARENA — HIGHLIGHT REEL")
    print(f"Backend: {backend}  |  Rounds: {num_rounds}  |  Seed: {seed}\n")
    time.sleep(pause_s)

    seen_convergence_count = 0

    try:
        for round_num in range(num_rounds):
            alive = [aid for aid, a in sim.world.agents.items() if a.alive]
            if len(alive) < 2:
                print(f"\n{DIM}Only {len(alive)} agent(s) remain. Ending highlight reel early.{RESET}")
                break

            await sim._run_round(round_num)

            if clear:
                print(CLEAR_SCREEN, end="")
            print(f"{BOLD}─── Round {round_num} / {num_rounds - 1} ───{RESET}\n")
            print(sim.world.render_ascii())

            print(f"\n{BOLD}Vocabulary so far:{RESET}")
            print(sim.symbol_tracker.get_vocabulary_table())

            # Highlight any NEW convergence events from this round in green
            new_events = sim.symbol_tracker.convergence_events[seen_convergence_count:]
            if new_events:
                print(f"\n{GREEN}{BOLD}*** WORD EMERGED THIS ROUND ***{RESET}")
                for evt in new_events:
                    print(
                        f"{GREEN}  '{evt['symbol']}' \u2192 {evt['context']} "
                        f"(adopted by {', '.join(evt['adopters'])}, stability {evt['stability']:.2f}){RESET}"
                    )
                seen_convergence_count = len(sim.symbol_tracker.convergence_events)

            bench = sim.benchmark.summary()
            if bench:
                print(
                    f"\n{DIM}Parallel inference: {bench['avg_parallel_inference_ms']:.0f}ms avg "
                    f"| est. sequential: {bench['avg_sequential_estimate_ms']:.0f}ms "
                    f"| speedup: {bench['parallel_speedup_factor']}\u00d7{RESET}"
                )

            time.sleep(pause_s)

    finally:
        await sim.agent_pool.close()
        sim.save_outputs()

    # ── Final wrap-up ──
    if clear:
        print(CLEAR_SCREEN, end="")
    _banner("FINAL RESULTS")

    print(f"{BOLD}Scores:{RESET}")
    for agent in sim.world.agents.values():
        status = f"{GREEN}alive{RESET}" if agent.alive else f"{DIM}DEAD{RESET}"
        print(f"  {agent.agent_id}: score={agent.score:.1f} "
              f"trades={agent.trades_completed}/{agent.trades_attempted} [{status}]")

    print(f"\n{BOLD}Final emergent vocabulary:{RESET}")
    print(sim.symbol_tracker.get_vocabulary_table())

    conv = sim.symbol_tracker.convergence_events
    if conv:
        print(f"\n{GREEN}{BOLD}Words that emerged this run:{RESET}")
        for evt in conv:
            print(f"{GREEN}  '{evt['symbol']}' \u2192 {evt['context']} "
                  f"(round {evt['round']}, adopters: {', '.join(evt['adopters'])}){RESET}")

    bench = sim.benchmark.summary()
    if bench:
        print(f"\n{BOLD}Benchmark:{RESET}")
        print(f"  Parallel inference (3 agents): {bench['avg_parallel_inference_ms']:.0f}ms avg/round")
        print(f"  Sequential estimate:           {bench['avg_sequential_estimate_ms']:.0f}ms avg/round")
        print(f"  Speedup:                        {bench['parallel_speedup_factor']}\u00d7")

    # Render the shareable benchmark card + vocabulary lineage tree as a wrap-up artefact
    try:
        from ui.benchmark_card import generate_benchmark_card
        bench_path = os.path.join(output_dir, "benchmark.json")
        if os.path.exists(bench_path):
            card_path = generate_benchmark_card(bench_path, os.path.join(output_dir, "benchmark_card.png"))
            print(f"\n{BOLD}Benchmark card saved:{RESET} {card_path}")
    except Exception as e:
        print(f"\n{DIM}(skipped benchmark card: {e}){RESET}")

    try:
        from ui.vocabulary_viz import generate_vocab_html
        vocab_path = os.path.join(output_dir, "vocabulary.json")
        if os.path.exists(vocab_path):
            viz_path = generate_vocab_html(vocab_path, os.path.join(output_dir, "vocabulary_viz.html"))
            print(f"{BOLD}Vocabulary lineage tree saved:{RESET} {viz_path}")
    except Exception as e:
        print(f"{DIM}(skipped vocabulary viz: {e}){RESET}")

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}{'END OF HIGHLIGHT REEL'.center(72)}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="30-round demo highlight reel for screen recording")
    parser.add_argument("--rounds", type=int, default=30, help="Number of rounds (default: 30)")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between rounds in seconds (default: 0.5)")
    parser.add_argument("--backend", choices=["auto", "fireworks", "heuristic"],
                        default="auto",
                        help="LLM backend (default: auto — picks the best available; "
                             "heuristic needs no API key)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear the terminal between rounds (useful for scrollback recordings)")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_demo(
        num_rounds=args.rounds,
        pause_s=args.pause,
        backend=args.backend,
        seed=args.seed,
        output_dir=args.output_dir,
        clear=not args.no_clear,
    ))


if __name__ == "__main__":
    main()
