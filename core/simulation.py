"""
simulation.py — Emergent Negotiation Arena

Orchestrates the full simulation loop:
  Round N:
    1. Get observations for all alive agents
    2. All agents decide IN PARALLEL (one API round-trip for all three)
    3. Resolve moves: agents collect resources
    4. Resolve trade proposals: match proposer → target → get target's response
    5. Execute accepted trades, apply rewards/penalties
    6. Update symbol tracker
    7. End round: deplete resources, apply survival scoring, respawn tiles
    8. Log everything for visualisation

Run for N rounds. Output:
  - simulation_log.json  — full event log
  - vocabulary.json      — discovered symbol vocabulary
  - benchmark.json       — latency benchmarks
"""

import asyncio
import json
import os
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

from core.grid_world import GridWorld, TradeProposal, RESOURCES, valid_resource_dict
from core.symbol_tracker import SymbolTracker
from agents.llm_agent import AgentPool


# ─────────────────────────────────────────────
# Benchmark recorder
# ─────────────────────────────────────────────

@dataclass
class RoundBenchmark:
    round_number: int
    parallel_inference_ms: float   # time for all 3 agents to decide (parallel)
    sequential_estimate_ms: float  # estimated sequential time (parallel × 3)
    trades_attempted: int
    trades_succeeded: int
    new_symbols: int
    alive_agents: int


class BenchmarkRecorder:
    def __init__(self):
        self.records: List[RoundBenchmark] = []

    def record(self, record: RoundBenchmark):
        self.records.append(record)

    def summary(self) -> dict:
        if not self.records:
            return {}
        total_parallel = sum(r.parallel_inference_ms for r in self.records)
        total_sequential = sum(r.sequential_estimate_ms for r in self.records)
        return {
            "total_rounds": len(self.records),
            "avg_parallel_inference_ms": round(total_parallel / len(self.records), 1),
            "avg_sequential_estimate_ms": round(total_sequential / len(self.records), 1),
            "parallel_speedup_factor": round(total_sequential / max(total_parallel, 1), 2),
            "total_trades_attempted": sum(r.trades_attempted for r in self.records),
            "total_trades_succeeded": sum(r.trades_succeeded for r in self.records),
            "total_symbols_invented": self.records[-1].new_symbols if self.records else 0,
            "note": "Sequential estimate = parallel_time × 3 (if agents ran one-at-a-time)"
        }

    def to_json(self, path: str = "benchmark.json"):
        data = {
            "summary": self.summary(),
            "per_round": [asdict(r) for r in self.records],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data


# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

class Simulation:
    """
    Full simulation orchestrator.

    Usage:
        sim = Simulation(num_rounds=100, verbose=True)
        asyncio.run(sim.run())
        sim.save_outputs()
    """

    def __init__(
        self,
        num_rounds: int = 100,
        seed: int = 42,
        verbose: bool = True,
        output_dir: str = ".",
        agent_pool=None,
    ):
        self.num_rounds = num_rounds
        self.seed = seed
        self.verbose = verbose
        self.output_dir = output_dir

        self.world = GridWorld(seed=seed)
        self.symbol_tracker = SymbolTracker()
        # An injected pool lets replay mode feed recorded decisions through
        # the exact same machinery, with zero LLM calls.
        self.agent_pool = agent_pool if agent_pool is not None else AgentPool()
        self.benchmark = BenchmarkRecorder()

        # Pending trade proposals: target_id → TradeProposal
        # Only one pending proposal per agent at a time
        self.pending_proposals: Dict[str, TradeProposal] = {}

        self.simulation_log: List[dict] = []

        # Immutable per-round snapshot for cross-thread readers (Gradio UI).
        # Swapped atomically at the end of each round; readers must never
        # iterate the live world/tracker structures from another thread.
        self.latest_snapshot: Optional[dict] = None

    @property
    def backend_name(self) -> str:
        return getattr(self.agent_pool, "backend_name", "unknown")

    # ── Main loop ──

    async def run(self):
        print(f"\n{'='*60}")
        print("EMERGENT NEGOTIATION ARENA")
        print(f"{'='*60}")
        print(f"Agents: 3 | Rounds: {self.num_rounds} | Backend: {self.backend_name}")
        print(f"{'='*60}\n")

        for round_num in range(self.num_rounds):
            alive = [aid for aid, a in self.world.agents.items() if a.alive]
            if len(alive) < 2:
                print(f"\n[Round {round_num}] Only {len(alive)} agent(s) remain. Simulation ending.")
                break

            await self._run_round(round_num)

            if self.verbose and round_num % 10 == 0:
                self._print_status(round_num)

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        self._print_final_summary()

    async def _run_round(self, round_num: int):
        round_start = time.perf_counter()

        # 1. Get observations
        observations = self.world.get_all_observations()

        # 2. All agents decide IN PARALLEL
        inference_start = time.perf_counter()
        decisions = await self.agent_pool.decide_all_parallel(
            observations,
            pending_proposals=self.pending_proposals,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000
        # Replay pools report the original run's logged latency instead of the
        # near-zero time it takes to look up a recorded decision.
        logged_ms = getattr(self.agent_pool, "last_inference_ms", None)
        if logged_ms is not None:
            inference_ms = logged_ms

        # 3. Resolve moves
        for agent_id, decision in decisions.items():
            if decision.get("action") == "move":
                direction = decision.get("direction", "stay")
                self.world.move_agent(agent_id, direction)

        # 4. Resolve trade proposals
        new_proposals: Dict[str, TradeProposal] = {}
        trade_results_this_round = []

        for agent_id, decision in decisions.items():
            action = decision.get("action")

            if action == "propose_trade":
                target_id = decision.get("target_id")
                offer = decision.get("offer", {})
                request = decision.get("request", {})
                symbol = decision.get("symbol", "?")

                # Validate target exists and is alive
                if not (target_id in self.world.agents
                        and self.world.agents[target_id].alive
                        and target_id != agent_id):
                    continue

                # Strict semantic validation — LLM output is untrusted input.
                # Invalid proposals are dropped visibly, never queued.
                if not (valid_resource_dict(offer) and valid_resource_dict(request)):
                    trade_results_this_round.append({
                        "round": round_num,
                        "type": "dropped_invalid",
                        "proposer": agent_id,
                        "target": target_id,
                        "offer": offer,
                        "request": request,
                        "symbol": symbol,
                    })
                    continue

                # One pending proposal per target: a second proposal to the
                # same target this round is dropped visibly instead of
                # silently overwriting the first (the target never sees a
                # dropped proposal, so its symbol is not registered either).
                if target_id in new_proposals:
                    trade_results_this_round.append({
                        "round": round_num,
                        "type": "dropped_collision",
                        "proposer": agent_id,
                        "target": target_id,
                        "symbol": symbol,
                    })
                    continue

                proposal = TradeProposal(
                    proposer_id=agent_id,
                    target_id=target_id,
                    offer=offer,
                    request=request,
                    symbol=symbol,
                    round_number=round_num,
                )
                # Register symbol before we know outcome
                self.symbol_tracker.record_proposal_symbol(
                    symbol=symbol,
                    proposer_id=agent_id,
                    target_id=target_id,
                    offer=offer,
                    request=request,
                    round_number=round_num,
                )
                # Queue for target's response
                new_proposals[target_id] = proposal

            elif action == "accept_trade":
                pending = self.pending_proposals.get(agent_id)
                if pending:
                    response_symbol = decision.get("symbol", "YES")
                    result = self.world.execute_trade(pending, accepted=True)
                    # The world is the source of truth: the agent agreeing
                    # does not mean the trade executed — resource shortfalls
                    # or invalid amounts fail inside execute_trade, and that
                    # failure must reach the tracker and both agents' memory.
                    actual_success = bool(result.accepted)
                    context = f"{list(pending.offer.keys())}→{list(pending.request.keys())}"
                    self.symbol_tracker.record_trade_outcome(
                        pending.symbol, accepted=actual_success, responder_id=agent_id
                    )
                    if actual_success:
                        # Accepting a trade under this symbol is the moment
                        # the responder commits to the convention.
                        self.symbol_tracker.record_response_symbol(
                            pending.symbol, agent_id, round_num
                        )
                    self.agent_pool.notify_trade_result(
                        pending.proposer_id, agent_id, pending.symbol,
                        actual_success, context
                    )
                    trade_results_this_round.append({
                        "round": round_num,
                        "type": "accepted" if actual_success else "failed",
                        "proposer": pending.proposer_id,
                        "target": agent_id,
                        "offer": pending.offer,
                        "request": pending.request,
                        "symbol": pending.symbol,
                        "response_symbol": response_symbol,
                    })

            elif action == "reject_trade":
                pending = self.pending_proposals.get(agent_id)
                if pending:
                    rejection_symbol = decision.get("symbol", "NO")
                    self.world.execute_trade(pending, accepted=False,
                                             rejection_symbol=rejection_symbol)
                    context = f"{list(pending.offer.keys())}→{list(pending.request.keys())}"
                    self.symbol_tracker.record_trade_outcome(
                        pending.symbol, accepted=False,
                        rejection_symbol=rejection_symbol, responder_id=agent_id
                    )
                    self.agent_pool.notify_trade_result(
                        pending.proposer_id, agent_id, pending.symbol, False, context
                    )
                    trade_results_this_round.append({
                        "round": round_num,
                        "type": "rejected",
                        "proposer": pending.proposer_id,
                        "target": agent_id,
                        "symbol": pending.symbol,
                        "rejection_symbol": rejection_symbol,
                    })

        # Update pending proposals for next round
        self.pending_proposals = new_proposals

        # 5. End round
        self.world.end_round()
        self.symbol_tracker.end_round(round_num)

        # 6. Benchmark record
        inference_ms_round = inference_ms
        self.benchmark.record(RoundBenchmark(
            round_number=round_num,
            parallel_inference_ms=inference_ms_round,
            sequential_estimate_ms=inference_ms_round * 3,
            trades_attempted=len([d for d in decisions.values()
                                  if d.get("action") == "propose_trade"]),
            trades_succeeded=len([t for t in trade_results_this_round
                                  if t.get("type") == "accepted"]),
            new_symbols=len(self.symbol_tracker.vocabulary),
            alive_agents=sum(1 for a in self.world.agents.values() if a.alive),
        ))

        # 7. Log
        round_log = {
            "round": round_num,
            "decisions": decisions,
            "trades": trade_results_this_round,
            "world_state": self.world.get_state_summary(),
            "symbol_summary": self.symbol_tracker.get_summary(),
            "inference_ms": round(inference_ms_round, 1),
        }
        self.simulation_log.append(round_log)

        # 8. Publish an immutable snapshot for cross-thread readers (UI)
        self._publish_snapshot()

    def _publish_snapshot(self):
        """
        Build a self-contained snapshot of everything the dashboard displays.

        The Gradio UI polls from a different thread than the simulation loop;
        iterating live dicts (e.g. the growing vocabulary) from that thread
        can raise "dict changed size during iteration". Readers therefore
        consume only this dict, which is swapped in atomically.
        """
        agents = [
            {
                "agent_id": a.agent_id,
                "alive": a.alive,
                "score": round(a.score, 2),
                "trades_completed": a.trades_completed,
                "trades_attempted": a.trades_attempted,
                "resources": dict(a.resources),
                "position": list(a.position),
            }
            for a in self.world.agents.values()
        ]
        tiles = [
            {"type": t.resource_type, "pos": list(t.position), "amount": t.amount}
            for t in self.world.resource_tiles
        ]
        vocab_entries = [e.to_dict() for e in self.symbol_tracker.vocabulary.values()]
        # Last 12 trade EVENTS across the whole run — windowing by round would
        # leave the final dashboard empty whenever trading went quiet late.
        recent_trades = [
            dict(t)
            for round_log in self.simulation_log
            for t in round_log.get("trades", [])
        ][-12:]
        self.latest_snapshot = {
            "round": self.world.round,
            "backend": self.backend_name,
            "alive_count": sum(1 for a in self.world.agents.values() if a.alive),
            "grid_ascii": self.world.render_ascii(),
            "agents": agents,
            "tiles": tiles,
            "vocab_table": self.symbol_tracker.get_vocabulary_table(),
            "vocab_entries": vocab_entries,
            "convergence_events": [dict(e) for e in self.symbol_tracker.convergence_events],
            "benchmark": self.benchmark.summary(),
            "recent_trades": recent_trades,
            # Per-round series for the activity chart (last 96 bars fit the panel)
            "history": [
                {"round": r.round_number, "attempted": r.trades_attempted,
                 "succeeded": r.trades_succeeded, "symbols": r.new_symbols,
                 "alive": r.alive_agents}
                for r in self.benchmark.records[-96:]
            ],
        }

    # ── Output ──

    def save_outputs(self):
        """Save all artefacts for submission and HF Space."""
        os.makedirs(self.output_dir, exist_ok=True)

        vocab_path = f"{self.output_dir}/vocabulary.json"
        bench_path = f"{self.output_dir}/benchmark.json"
        log_path = f"{self.output_dir}/simulation_log.json"

        self.symbol_tracker.to_json(vocab_path)
        self.benchmark.to_json(bench_path)

        # The log carries its own metadata so any saved run can later be
        # replayed exactly (core/replay.py needs the seed to rebuild the
        # identical world) and so benchmark claims stay attributable.
        log_payload = {
            "meta": {
                "backend": self.backend_name,
                "seed": self.seed,
                "rounds_completed": len(self.simulation_log),
                "rounds_requested": self.num_rounds,
                "semantics_mode": os.getenv("SEMANTICS_MODE", "visible"),
                "generated_unix_time": time.time(),
            },
            "rounds": self.simulation_log,
        }
        with open(log_path, "w") as f:
            json.dump(log_payload, f, indent=2)

        print(f"\nOutputs saved:")
        print(f"  {vocab_path}")
        print(f"  {bench_path}")
        print(f"  {log_path}")

    # ── Display ──

    def _print_status(self, round_num: int):
        print(f"\n{'─'*50}")
        print(f"Round {round_num}")
        print(self.world.render_ascii())
        print("\nVocabulary so far:")
        print(self.symbol_tracker.get_vocabulary_table())
        bench = self.benchmark.summary()
        if bench:
            print(f"\nParallel inference: {bench['avg_parallel_inference_ms']:.0f}ms avg "
                  f"| Est. sequential: {bench['avg_sequential_estimate_ms']:.0f}ms "
                  f"| Speedup: {bench['parallel_speedup_factor']}×")

    def _print_final_summary(self):
        print("\nFINAL SCORES:")
        for agent in self.world.agents.values():
            status = "alive" if agent.alive else "DEAD"
            print(f"  {agent.agent_id}: score={agent.score:.1f} "
                  f"trades={agent.trades_completed}/{agent.trades_attempted} "
                  f"resources={dict(agent.resources)} [{status}]")

        print("\nEMERGENT VOCABULARY:")
        print(self.symbol_tracker.get_vocabulary_table())

        bench = self.benchmark.summary()
        if bench:
            print(f"\nBENCHMARK:")
            print(f"  Parallel inference (3 agents): {bench['avg_parallel_inference_ms']:.0f}ms avg per round")
            print(f"  Sequential estimate: {bench['avg_sequential_estimate_ms']:.0f}ms avg per round")
            print(f"  Speedup from parallelism: {bench['parallel_speedup_factor']}×")
            print(f"  Total trades: {bench['total_trades_attempted']} attempted, "
                  f"{bench['total_trades_succeeded']} succeeded")
            print(f"  Symbols invented: {bench['total_symbols_invented']}")

        conv = self.symbol_tracker.convergence_events
        if conv:
            print(f"\nCONVERGENCE EVENTS (words that emerged):")
            for evt in conv[:5]:
                print(f"  '{evt['symbol']}' → {evt['context']} "
                      f"(adopted by {evt['adopters']}, round {evt['round']})")
