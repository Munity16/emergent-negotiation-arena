"""
replay.py — Emergent Negotiation Arena

Deterministic playback of a recorded simulation — the guaranteed judging /
demo path when no GPU, no API key, and no network are available.

A saved simulation log (outputs/simulation_log.json) contains every agent
decision per round plus the seed it ran with. Because the world evolves
deterministically from (seed, decisions), feeding the logged decisions back
through the real Simulation machinery reproduces the entire run — grid,
trades, vocabulary, convergence events, benchmark — with zero LLM calls.

Usage:
    from core.replay import load_replay_file, ScriptedAgentPool
    meta, rounds = load_replay_file("outputs/sample_run.json")
    pool = ScriptedAgentPool(rounds, playback_delay_s=0.3)
    sim = Simulation(num_rounds=len(rounds), seed=meta["seed"], agent_pool=pool)
    asyncio.run(sim.run())
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple


class ReplayFormatError(ValueError):
    """The given file is not a replayable simulation log."""


def load_replay_file(path: str) -> Tuple[dict, List[dict]]:
    """
    Load a saved simulation log. Returns (meta, rounds).

    Accepts the current {"meta": ..., "rounds": [...]} format and the legacy
    bare-list format (meta comes back empty — the caller must supply a seed).
    """
    if not os.path.exists(path):
        raise ReplayFormatError(f"Replay file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "rounds" in data:
        meta = data.get("meta") or {}
        rounds = data["rounds"]
    elif isinstance(data, list):
        meta, rounds = {}, data
    else:
        raise ReplayFormatError(f"{path} is not a simulation log")

    if (not isinstance(rounds, list) or not rounds
            or not isinstance(rounds[0], dict) or "decisions" not in rounds[0]):
        raise ReplayFormatError(f"{path} contains no per-round decisions to replay")
    return meta, rounds


class ScriptedAgentPool:
    """
    Duck-typed stand-in for agents.llm_agent.AgentPool that replays logged
    decisions instead of calling an LLM. Simulation treats it identically.
    """

    def __init__(self, rounds: List[dict], playback_delay_s: float = 0.0):
        self._rounds = rounds
        self._idx = 0
        # Slows playback so a human (or screen recording) can watch the run
        # unfold; 0 for tests.
        self.playback_delay_s = playback_delay_s
        # Simulation displays the ORIGINAL run's logged latency, not the
        # near-zero time it takes to look up a recorded decision.
        self.last_inference_ms: Optional[float] = 0.0
        self.backend_name = "replay"

    async def decide_all_parallel(
        self,
        observations: Dict[str, dict],
        pending_proposals=None,
    ) -> Dict[str, dict]:
        if self.playback_delay_s > 0:
            await asyncio.sleep(self.playback_delay_s)

        stay = {"action": "move", "direction": "stay", "reasoning": "replay-pad"}
        if self._idx >= len(self._rounds):
            self.last_inference_ms = 0.0
            return {aid: dict(stay) for aid in observations}

        entry = self._rounds[self._idx]
        self._idx += 1
        self.last_inference_ms = float(entry.get("inference_ms") or 0.0)
        decisions = entry.get("decisions", {})
        return {aid: decisions.get(aid, dict(stay)) for aid in observations}

    def notify_trade_result(self, *args, **kwargs):
        pass  # outcomes are already baked into the recorded decisions

    async def close(self):
        pass
