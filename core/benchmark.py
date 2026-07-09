"""
core/benchmark.py — Emergent Negotiation Arena

Standalone parallel-vs-sequential inference benchmark.

Runs N rounds (default 10) of the 3-agent decision step twice per round:
  - PARALLEL   : asyncio.gather() over all 3 agents — one API round-trip
                 covers all three decisions
  - SEQUENTIAL : the same 3 decisions made one-at-a-time ("before" baseline)

Writes `benchmark_llm.json` with per-round timings and a summary
(headline speedup number, avg latencies).

Usage:
    python core/benchmark.py --rounds 10
    python core/benchmark.py --rounds 10 --mode live       # force real API calls
    python core/benchmark.py --rounds 10 --mode simulate   # force synthetic timings
    python core/benchmark.py --rounds 10 --output outputs/benchmark_llm.json

Modes:
    auto     (default) — try a real call to the configured backend; if it's
               unreachable, fall back to a synthetic timing model and label
               the output as such.
    live     — always call the real backend (fireworks, needs FIREWORKS_API_KEY).
               Will error out if the backend is unreachable.
    simulate — never touch the network; use a synthetic per-call latency model
               calibrated to typical request latency (~55-90ms/call).
"""

import argparse
import asyncio
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.grid_world import GridWorld
from agents.llm_agent import AgentPool, LLMClient, NegotiationAgent

AGENT_IDS = ["agent_0", "agent_1", "agent_2"]

# Synthetic latency model (ms), used only in "simulate" mode or as an "auto" fallback.
# Calibrated to roughly match a typical LLM request for a ~500-token
# negotiation prompt: base cost + small jitter, independent per call.
SIMULATED_CALL_MS_MEAN = 62.0
SIMULATED_CALL_MS_JITTER = 18.0


@dataclass
class RoundTiming:
    round_number: int
    parallel_ms: float
    sequential_ms: float
    speedup: float


class SimulatedAgent:
    """Drop-in stand-in for NegotiationAgent.decide() when no live backend is reachable."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def decide(self, observation: dict, pending_proposal=None) -> dict:
        latency_s = max(
            0.001,
            random.gauss(SIMULATED_CALL_MS_MEAN, SIMULATED_CALL_MS_JITTER) / 1000.0,
        )
        await asyncio.sleep(latency_s)
        return {"action": "move", "direction": random.choice(
            ["up", "down", "left", "right", "stay"]), "reasoning": "simulated"}


async def _probe_live_backend(pool: AgentPool, observation: dict, timeout_s: float = 5.0) -> bool:
    """Fire one quick call to see if the configured LLM backend is reachable."""
    try:
        await asyncio.wait_for(
            pool.agents[AGENT_IDS[0]].llm.complete("ping", "Respond with {\"action\": \"move\", \"direction\": \"stay\"}"),
            timeout=timeout_s,
        )
        return True
    except Exception:
        return False


async def run_benchmark(num_rounds: int, mode: str, output_path: str) -> dict:
    world = GridWorld(seed=7)
    pool = AgentPool()

    resolved_mode = mode
    if mode == "auto":
        print("Probing live backend...")
        obs_probe = world.get_observation(AGENT_IDS[0])
        reachable = await _probe_live_backend(pool, obs_probe)
        resolved_mode = "live" if reachable else "simulate"
        print(f"  -> backend reachable: {reachable}. Using mode='{resolved_mode}'.")

    if resolved_mode == "simulate":
        agents = {aid: SimulatedAgent(aid) for aid in AGENT_IDS}
    else:
        agents = pool.agents  # real NegotiationAgent instances hitting the real backend

    timings: List[RoundTiming] = []

    for round_num in range(num_rounds):
        observations = world.get_all_observations()
        if len(observations) < 3:
            # keep the benchmark deterministic/full even if agents "died" from
            # prior rounds' resource depletion inside this standalone script
            observations = {aid: world.get_observation(aid) for aid in AGENT_IDS}

        # ── PARALLEL ──
        t0 = time.perf_counter()
        await asyncio.gather(*[
            agents[aid].decide(observations[aid]) for aid in AGENT_IDS
        ])
        parallel_ms = (time.perf_counter() - t0) * 1000

        # ── SEQUENTIAL (the "before" baseline) ──
        t0 = time.perf_counter()
        for aid in AGENT_IDS:
            await agents[aid].decide(observations[aid])
        sequential_ms = (time.perf_counter() - t0) * 1000

        speedup = round(sequential_ms / max(parallel_ms, 1e-6), 3)
        timings.append(RoundTiming(
            round_number=round_num,
            parallel_ms=round(parallel_ms, 2),
            sequential_ms=round(sequential_ms, 2),
            speedup=speedup,
        ))
        print(f"  round {round_num:>2} | parallel {parallel_ms:7.1f}ms "
              f"| sequential {sequential_ms:7.1f}ms | speedup {speedup:.2f}x")

        # advance the world a little so later rounds have varied observations
        world.end_round()

    if resolved_mode != "simulate":
        await pool.close()

    avg_parallel = sum(t.parallel_ms for t in timings) / len(timings)
    avg_sequential = sum(t.sequential_ms for t in timings) / len(timings)
    overall_speedup = round(avg_sequential / max(avg_parallel, 1e-6), 3)

    result = {
        "meta": {
            "benchmark": "emergent-negotiation-arena",
            "backend": "Fireworks AI API (parallel asyncio.gather)",
            "model": os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/gpt-oss-120b"),
            "agents": len(AGENT_IDS),
            "rounds": num_rounds,
            "mode": resolved_mode,
            "note": (
                "mode='live' timings were measured against a real LLM backend."
                if resolved_mode == "live" else
                "mode='simulate': no live LLM backend was reachable from this "
                "environment, so timings use a synthetic latency model calibrated "
                "to Qwen3-1.7B request latency. Re-run with --mode live and a "
                "FIREWORKS_API_KEY for measured numbers."
            ),
            "host": platform.platform(),
            "generated_unix_time": time.time(),
        },
        "before_after": {
            "before_sequential_avg_ms": round(avg_sequential, 1),
            "after_parallel_avg_ms": round(avg_parallel, 1),
            "speedup_factor": overall_speedup,
            "headline": f"{overall_speedup:.2f}x speedup: {round(avg_sequential):d}ms -> {round(avg_parallel):d}ms per round",
        },
        "per_round": [asdict(t) for t in timings],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE ({resolved_mode} mode)")
    print(f"{'='*60}")
    print(f"  Before (sequential, avg/round): {avg_sequential:7.1f} ms")
    print(f"  After  (parallel,   avg/round): {avg_parallel:7.1f} ms")
    print(f"  Speedup: {overall_speedup:.2f}x")
    print(f"  Saved to: {output_path}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel-vs-sequential LLM inference benchmark")
    parser.add_argument("--rounds", type=int, default=10, help="Number of benchmark rounds (default: 10)")
    parser.add_argument("--mode", choices=["auto", "live", "simulate"], default="auto",
                        help="auto = probe backend and fall back if unreachable; "
                             "live = force real backend calls; simulate = synthetic timings only")
    parser.add_argument("--output", type=str, default="benchmark_llm.json",
                        help="Output JSON path (default: benchmark_llm.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args.rounds, args.mode, args.output))


if __name__ == "__main__":
    main()
