"""
tests/test_fixes.py — Emergent Negotiation Arena

Regression tests for the audit-report failure modes:
  - a failed "accepted" trade must be recorded as failed everywhere (P0-1)
  - negative / zero / non-integer / unknown resources are rejected (P0-2)
  - receiving a proposal does not make the target an adopter (P0-3)
  - appears_desperate lists only genuinely low resources (P0-4)
  - heuristic backend produces trades and symbols with no key (P0-5)
  - proposal collisions are visible, not silent overwrites
  - replay reproduces a recorded run exactly
  - backend resolution falls back cleanly when nothing live is available

Run:
    pytest tests/test_fixes.py -v
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.grid_world import GridWorld, TradeProposal, valid_resource_dict
from core.replay import ScriptedAgentPool, load_replay_file
from core.simulation import Simulation
from agents.llm_agent import BackendUnavailable, LLMClient, resolve_backend


# ─────────────────────────────────────────────
# Strict resource validation (P0-2)
# ─────────────────────────────────────────────

class TestResourceValidation:
    @pytest.mark.parametrize("bad", [
        {},                       # empty
        None,                     # not a dict
        [("food", 1)],            # not a dict
        {"food": -1},             # negative
        {"food": 0},              # zero
        {"food": 1.5},            # non-integer
        {"food": True},           # bool masquerading as int
        {"food": "2"},            # string amount
        {"gold": 1},              # unknown resource
        {"food": 2, "gold": 1},   # one bad key poisons the dict
    ])
    def test_invalid_dicts_rejected(self, bad):
        assert valid_resource_dict(bad) is False

    @pytest.mark.parametrize("good", [
        {"food": 1},
        {"food": 2, "water": 3},
        {"food": 1, "water": 1, "metal": 1, "energy": 1},
    ])
    def test_valid_dicts_accepted(self, good):
        assert valid_resource_dict(good) is True

    def test_negative_offer_cannot_steal_resources(self):
        """A negative amount used to reverse the transfer direction."""
        world = GridWorld(seed=42)
        before_proposer = dict(world.agents["agent_0"].resources)
        before_target = dict(world.agents["agent_1"].resources)

        proposal = TradeProposal(
            proposer_id="agent_0", target_id="agent_1",
            offer={"food": -3}, request={"water": 1}, symbol="EVIL",
        )
        result = world.execute_trade(proposal, accepted=True)

        assert result.accepted is False
        assert world.agents["agent_0"].resources == before_proposer
        assert world.agents["agent_1"].resources == before_target

    def test_unknown_resource_cannot_corrupt_or_crash(self):
        """An unknown resource name used to raise KeyError mid-transfer."""
        world = GridWorld(seed=42)
        proposal = TradeProposal(
            proposer_id="agent_0", target_id="agent_1",
            offer={"gold": 1}, request={"water": -1}, symbol="GOLD",
        )
        result = world.execute_trade(proposal, accepted=True)  # must not raise
        assert result.accepted is False
        assert "gold" not in world.agents["agent_0"].resources
        assert "gold" not in world.agents["agent_1"].resources


# ─────────────────────────────────────────────
# Observation truthfulness (P0-4)
# ─────────────────────────────────────────────

class TestAppearsDesperate:
    def test_only_actually_low_resources_listed(self):
        world = GridWorld(seed=42)
        # agent_1 is visible from agent_0 (distance 5) and low ONLY on water
        world.agents["agent_1"].resources = {"food": 8, "water": 1, "metal": 5, "energy": 5}
        obs = world.get_observation("agent_0")
        visible = {v["agent_id"]: v for v in obs["visible_agents"]}
        assert visible["agent_1"]["appears_desperate"] == ["water"]

    def test_empty_when_nothing_low(self):
        world = GridWorld(seed=42)
        world.agents["agent_1"].resources = {"food": 8, "water": 8, "metal": 8, "energy": 8}
        obs = world.get_observation("agent_0")
        visible = {v["agent_id"]: v for v in obs["visible_agents"]}
        assert visible["agent_1"]["appears_desperate"] == []


# ─────────────────────────────────────────────
# Simulation-level truthfulness (P0-1) and collisions
# ─────────────────────────────────────────────

def _run_scripted(rounds, num_rounds=None, seed=42):
    """Run a simulation from scripted per-round decisions; returns the sim."""
    script = [{"decisions": d, "inference_ms": 0.0} for d in rounds]
    pool = ScriptedAgentPool(script)
    sim = Simulation(num_rounds=num_rounds or len(rounds), seed=seed,
                     verbose=False, agent_pool=pool)
    asyncio.run(sim.run())
    return sim


class TestTradeOutcomeTruthfulness:
    def test_failed_execution_recorded_as_failed_everywhere(self):
        """Accepting a trade the target cannot pay must not count as success."""
        sim = _run_scripted([
            # round 0: agent_0 proposes an impossible request to agent_1
            {"agent_0": {"action": "propose_trade", "target_id": "agent_1",
                         "offer": {"food": 2}, "request": {"water": 99},
                         "symbol": "LIE"}},
            # round 1: agent_1 accepts — but execution must fail
            {"agent_1": {"action": "accept_trade", "symbol": "OK"}},
        ])

        entry = sim.symbol_tracker.vocabulary["LIE"]
        assert entry.success_count == 0
        assert entry.rejection_count == 1
        # the responder never committed to the convention
        assert entry.adopters == ["agent_0"]

        assert sim.world.get_state_summary()["successful_trades"] == 0
        assert sim.world.agents["agent_0"].trades_completed == 0
        assert sim.world.agents["agent_1"].trades_completed == 0

        trade_types = [t["type"] for r in sim.simulation_log for t in r["trades"]]
        assert "failed" in trade_types
        assert "accepted" not in trade_types

    def test_successful_trade_still_recorded_as_success(self):
        sim = _run_scripted([
            {"agent_0": {"action": "propose_trade", "target_id": "agent_1",
                         "offer": {"food": 2}, "request": {"water": 2},
                         "symbol": "ZRK"}},
            {"agent_1": {"action": "accept_trade", "symbol": "OK"}},
        ])
        entry = sim.symbol_tracker.vocabulary["ZRK"]
        assert entry.success_count == 1
        # acceptance = commitment -> responder is now an adopter -> convergence
        assert set(entry.adopters) == {"agent_0", "agent_1"}
        assert len(sim.symbol_tracker.convergence_events) == 1
        assert sim.world.get_state_summary()["successful_trades"] == 1

    def test_invalid_proposal_dropped_visibly_and_not_queued(self):
        sim = _run_scripted([
            {"agent_0": {"action": "propose_trade", "target_id": "agent_1",
                         "offer": {"food": -1}, "request": {"water": 1},
                         "symbol": "BAD"}},
            {"agent_1": {"action": "accept_trade", "symbol": "OK"}},
        ])
        trade_types = [t["type"] for r in sim.simulation_log for t in r["trades"]]
        assert "dropped_invalid" in trade_types
        # never queued, never registered, never executed
        assert "BAD" not in sim.symbol_tracker.vocabulary
        assert sim.world.get_state_summary()["total_trades"] == 0

    def test_proposal_collision_dropped_visibly(self):
        sim = _run_scripted([
            {
                "agent_0": {"action": "propose_trade", "target_id": "agent_1",
                            "offer": {"food": 1}, "request": {"water": 1},
                            "symbol": "FIRST"},
                "agent_2": {"action": "propose_trade", "target_id": "agent_1",
                            "offer": {"metal": 1}, "request": {"water": 1},
                            "symbol": "SECOND"},
            },
        ], num_rounds=1)
        collisions = [t for r in sim.simulation_log for t in r["trades"]
                      if t["type"] == "dropped_collision"]
        assert len(collisions) == 1
        assert collisions[0]["proposer"] == "agent_2"
        # the surviving pending proposal is the first one
        assert sim.pending_proposals["agent_1"].proposer_id == "agent_0"
        # a dropped proposal was never seen by anyone — not a symbol usage
        assert "SECOND" not in sim.symbol_tracker.vocabulary


# ─────────────────────────────────────────────
# Heuristic backend (P0-5): the no-key demo must actually demo
# ─────────────────────────────────────────────

class TestHeuristicBackend:
    def test_no_key_run_produces_trades_and_symbols(self, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "heuristic")
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

        sim = Simulation(num_rounds=40, seed=42, verbose=False)
        asyncio.run(sim.run())

        summary = sim.world.get_state_summary()
        assert summary["successful_trades"] > 0, "no-key demo produced zero trades"
        assert len(sim.symbol_tracker.vocabulary) > 0, "no-key demo produced zero symbols"
        # the arena must stay watchable, not collapse instantly
        assert sim.world.round >= 20

    def test_heuristic_is_deterministic(self, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "heuristic")

        def final_state():
            sim = Simulation(num_rounds=25, seed=42, verbose=False)
            asyncio.run(sim.run())
            return json.dumps(sim.world.get_state_summary(), sort_keys=True)

        assert final_state() == final_state()


# ─────────────────────────────────────────────
# Replay: recorded runs reproduce exactly
# ─────────────────────────────────────────────

class TestReplay:
    def test_replay_reproduces_recorded_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "heuristic")

        recorded = Simulation(num_rounds=15, seed=42, verbose=False)
        asyncio.run(recorded.run())
        recorded_summary = recorded.world.get_state_summary()

        log_file = tmp_path / "run.json"
        log_file.write_text(json.dumps({
            "meta": {"backend": "heuristic", "seed": 42},
            "rounds": recorded.simulation_log,
        }), encoding="utf-8")

        meta, rounds = load_replay_file(str(log_file))
        replayed = Simulation(num_rounds=len(rounds), seed=meta["seed"],
                              verbose=False, agent_pool=ScriptedAgentPool(rounds))
        asyncio.run(replayed.run())

        assert replayed.world.get_state_summary() == recorded_summary
        assert (len(replayed.symbol_tracker.vocabulary)
                == len(recorded.symbol_tracker.vocabulary))


# ─────────────────────────────────────────────
# Backend resolution and client config
# ─────────────────────────────────────────────

class TestBackendResolution:
    def test_auto_falls_back_to_heuristic_when_nothing_live(self, monkeypatch):
        # remove the Fireworks key; no replay file
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        backend, reason = resolve_backend("auto", replay_file=None)
        assert backend == "heuristic"
        assert "auto" in reason

    def test_auto_prefers_fireworks_when_key_present(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        backend, reason = resolve_backend("auto", replay_file=None)
        assert backend == "fireworks"
        assert "auto" in reason

    def test_explicit_fireworks_without_key_fails_fast(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(BackendUnavailable):
            resolve_backend("fireworks")

    def test_explicit_replay_without_file_fails_fast(self, tmp_path):
        with pytest.raises(BackendUnavailable):
            resolve_backend("replay", replay_file=str(tmp_path / "missing.json"))

    def test_llm_client_reads_config_at_instantiation(self, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "fireworks")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        monkeypatch.setenv("FIREWORKS_MODEL", "accounts/fireworks/models/other")
        client = LLMClient()
        assert client.backend == "fireworks"
        assert client.fireworks_model == "accounts/fireworks/models/other"
        asyncio.run(client.close())

    def test_llm_client_rejects_unknown_backend(self, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "vllm")  # removed backend
        with pytest.raises(ValueError):
            LLMClient()

    def test_fireworks_backend_without_key_raises_at_init(self, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "fireworks")
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            LLMClient()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
