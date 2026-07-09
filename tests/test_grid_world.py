"""
tests/test_grid_world.py — Emergent Negotiation Arena

Unit tests for core.grid_world.GridWorld:
  - agent initialisation
  - movement (valid + boundary-invalid)
  - resource collection
  - trade execution (valid, rejected, insufficient resources / lying)
  - survival scoring (penalty / bonus)
  - death condition

Run:
    pytest tests/test_grid_world.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.grid_world import (
    GridWorld,
    AgentState,
    ResourceTile,
    TradeProposal,
    RESOURCES,
    GRID_SIZE,
    SURVIVAL_THRESHOLD,
    PENALTY_ZERO,
    REWARD_THRESHOLD,
    REWARD_TRADE_SUCCESS,
    PENALTY_FAILED_TRADE,
    AGENT_START_POSITIONS,
    AGENT_START_RESOURCES,
    COLLECT_PER_VISIT,
    RESPAWN_ROUNDS,
)


@pytest.fixture
def world():
    return GridWorld(seed=42)


# ─────────────────────────────────────────────
# Agent initialisation
# ─────────────────────────────────────────────

class TestAgentInit:
    def test_three_agents_created(self, world):
        assert len(world.agents) == 3
        assert set(world.agents.keys()) == {"agent_0", "agent_1", "agent_2"}

    def test_agent_starting_positions_match_config(self, world):
        for agent_id, pos in AGENT_START_POSITIONS.items():
            assert world.agents[agent_id].position == pos

    def test_agent_starting_resources_match_config(self, world):
        for agent_id, resources in AGENT_START_RESOURCES.items():
            assert world.agents[agent_id].resources == resources

    def test_agents_start_alive_with_zero_score(self, world):
        for agent in world.agents.values():
            assert agent.alive is True
            assert agent.score == 0.0
            assert agent.trades_completed == 0
            assert agent.trades_attempted == 0

    def test_resource_tiles_spawned(self, world):
        assert len(world.resource_tiles) > 0
        types_present = {t.resource_type for t in world.resource_tiles}
        assert types_present == set(RESOURCES)

    def test_round_starts_at_zero(self, world):
        assert world.round == 0

    def test_is_desperate_reflects_threshold(self, world):
        # Set resources explicitly rather than relying on AGENT_START_RESOURCES,
        # so this test stays valid regardless of how the starting economy is tuned.
        agent = world.agents["agent_0"]
        agent.resources = {"food": 8, "water": 1, "metal": 1, "energy": 1}
        desperate = agent.is_desperate()
        # water/metal/energy = 1, below SURVIVAL_THRESHOLD (3)
        assert desperate["water"] is True
        assert desperate["metal"] is True
        assert desperate["energy"] is True
        # food = 8, above threshold
        assert desperate["food"] is False

    def test_surplus_reflects_threshold(self, world):
        agent = world.agents["agent_0"]
        agent.resources = {"food": 8, "water": SURVIVAL_THRESHOLD, "metal": 1, "energy": 1}
        surplus = agent.surplus()
        assert surplus["food"] == 8 - SURVIVAL_THRESHOLD
        assert surplus["water"] == 0


# ─────────────────────────────────────────────
# Movement
# ─────────────────────────────────────────────

class TestMoveAgent:
    def test_valid_move_updates_position(self, world):
        start = world.agents["agent_0"].position
        moved = world.move_agent("agent_0", "right")
        assert moved is True
        r, c = start
        assert world.agents["agent_0"].position == (r, c + 1)

    def test_stay_does_not_change_position(self, world):
        start = world.agents["agent_0"].position
        world.move_agent("agent_0", "stay")
        assert world.agents["agent_0"].position == start

    def test_unknown_direction_defaults_to_stay(self, world):
        start = world.agents["agent_0"].position
        moved = world.move_agent("agent_0", "diagonal")
        assert moved is True
        assert world.agents["agent_0"].position == start

    def test_move_out_of_bounds_top_left_is_rejected(self, world):
        agent = world.agents["agent_0"]
        agent.position = (0, 0)
        moved_up = world.move_agent("agent_0", "up")
        assert moved_up is False
        assert agent.position == (0, 0)
        moved_left = world.move_agent("agent_0", "left")
        assert moved_left is False
        assert agent.position == (0, 0)

    def test_move_out_of_bounds_bottom_right_is_rejected(self, world):
        agent = world.agents["agent_1"]
        agent.position = (GRID_SIZE - 1, GRID_SIZE - 1)
        moved_down = world.move_agent("agent_1", "down")
        assert moved_down is False
        moved_right = world.move_agent("agent_1", "right")
        assert moved_right is False
        assert agent.position == (GRID_SIZE - 1, GRID_SIZE - 1)


# ─────────────────────────────────────────────
# Resource collection
# ─────────────────────────────────────────────

class TestResourceCollection:
    def test_landing_on_tile_collects_resource(self, world):
        agent = world.agents["agent_0"]
        # Force a known tile next to the agent and move onto it
        r, c = agent.position
        target_pos = (r, c + 1)
        world.resource_tiles = [ResourceTile(resource_type="food", position=target_pos, amount=5)]

        before = agent.resources["food"]
        world.move_agent("agent_0", "right")
        after = agent.resources["food"]

        assert after == before + COLLECT_PER_VISIT  # capped per visit
        assert world.resource_tiles[0].amount == 5 - COLLECT_PER_VISIT

    def test_collection_capped_by_tile_amount(self, world):
        agent = world.agents["agent_0"]
        r, c = agent.position
        target_pos = (r, c + 1)
        world.resource_tiles = [ResourceTile(resource_type="food", position=target_pos, amount=2)]

        before = agent.resources["food"]
        world.move_agent("agent_0", "right")
        assert agent.resources["food"] == before + 2  # only what the tile held

    def test_collecting_last_units_sets_respawn_timer(self, world):
        agent = world.agents["agent_0"]
        r, c = agent.position
        target_pos = (r, c + 1)
        world.resource_tiles = [ResourceTile(resource_type="water", position=target_pos, amount=2)]

        world.move_agent("agent_0", "right")

        tile = world.resource_tiles[0]
        assert tile.amount == 0
        assert tile.respawn_in == RESPAWN_ROUNDS

    def test_no_collection_from_empty_tile(self, world):
        agent = world.agents["agent_0"]
        r, c = agent.position
        target_pos = (r, c + 1)
        world.resource_tiles = [ResourceTile(resource_type="metal", position=target_pos, amount=0)]

        before = agent.resources["metal"]
        world.move_agent("agent_0", "right")
        assert agent.resources["metal"] == before


# ─────────────────────────────────────────────
# Trade execution
# ─────────────────────────────────────────────

class TestTradeExecution:
    def test_valid_accepted_trade_transfers_resources_and_rewards(self, world):
        proposer = world.agents["agent_0"]  # food-rich
        target = world.agents["agent_1"]    # water-rich

        # Set resources explicitly rather than relying on AGENT_START_RESOURCES,
        # so this test stays valid regardless of how the starting economy is tuned.
        proposer.resources = {"food": 8, "water": 1, "metal": 5, "energy": 5}
        target.resources = {"food": 1, "water": 8, "metal": 5, "energy": 5}

        proposal = TradeProposal(
            proposer_id="agent_0",
            target_id="agent_1",
            offer={"food": 2},
            request={"water": 2},
            symbol="ZRK",
            round_number=0,
        )

        proposer_food_before = proposer.resources["food"]
        target_water_before = target.resources["water"]

        result = world.execute_trade(proposal, accepted=True)

        assert result.accepted is True
        assert proposer.resources["food"] == proposer_food_before - 2
        assert proposer.resources["water"] == 1 + 2  # started with 1 water
        assert target.resources["water"] == target_water_before - 2
        assert target.resources["food"] == 1 + 2  # started with 1 food

        assert proposer.score == REWARD_TRADE_SUCCESS
        assert target.score == REWARD_TRADE_SUCCESS
        assert proposer.trades_completed == 1
        assert target.trades_completed == 1
        assert proposer.trades_attempted == 1
        assert len(world.trade_history) == 1

    def test_rejected_trade_penalises_proposer_only(self, world):
        proposer = world.agents["agent_0"]
        proposal = TradeProposal(
            proposer_id="agent_0",
            target_id="agent_1",
            offer={"food": 2},
            request={"water": 2},
            symbol="ZRK",
            round_number=0,
        )

        result = world.execute_trade(proposal, accepted=False, rejection_symbol="NO")

        assert result.accepted is False
        assert result.rejection_symbol == "NO"
        assert proposer.score == PENALTY_FAILED_TRADE
        assert proposer.trades_completed == 0
        # no resources should move
        assert proposer.resources["food"] == AGENT_START_RESOURCES["agent_0"]["food"]

    def test_trade_with_proposer_insufficient_resources_fails(self, world):
        proposer = world.agents["agent_0"]
        target = world.agents["agent_1"]

        proposal = TradeProposal(
            proposer_id="agent_0",
            target_id="agent_1",
            offer={"food": 999},  # proposer does not actually have this much
            request={"water": 2},
            symbol="LIE1",
            round_number=0,
        )

        proposer_food_before = proposer.resources["food"]
        target_water_before = target.resources["water"]

        result = world.execute_trade(proposal, accepted=True)

        assert result.accepted is False
        assert proposer.score == PENALTY_FAILED_TRADE
        # nothing transferred
        assert proposer.resources["food"] == proposer_food_before
        assert target.resources["water"] == target_water_before

    def test_trade_with_target_insufficient_resources_fails(self, world):
        proposer = world.agents["agent_0"]
        target = world.agents["agent_1"]

        proposal = TradeProposal(
            proposer_id="agent_0",
            target_id="agent_1",
            offer={"food": 2},
            request={"water": 999},  # target does not have this much
            symbol="LIE2",
            round_number=0,
        )

        result = world.execute_trade(proposal, accepted=True)

        assert result.accepted is False
        assert proposer.score == PENALTY_FAILED_TRADE
        assert proposer.resources["food"] == AGENT_START_RESOURCES["agent_0"]["food"]
        assert target.resources["water"] == AGENT_START_RESOURCES["agent_1"]["water"]

    def test_trades_attempted_increments_regardless_of_outcome(self, world):
        proposal = TradeProposal(
            proposer_id="agent_0",
            target_id="agent_1",
            offer={"food": 2},
            request={"water": 2},
            symbol="ZRK",
            round_number=0,
        )
        world.execute_trade(proposal, accepted=True)
        world.execute_trade(proposal, accepted=False)
        assert world.agents["agent_0"].trades_attempted == 2


# ─────────────────────────────────────────────
# Survival scoring
# ─────────────────────────────────────────────

class TestSurvivalScoring:
    def test_resources_deplete_each_round(self, world):
        agent = world.agents["agent_2"]
        before = dict(agent.resources)
        world.end_round()
        for r in RESOURCES:
            assert agent.resources[r] == max(0, before[r] - 1)

    def test_penalty_applied_when_resource_hits_zero(self, world):
        agent = world.agents["agent_1"]
        # Set resources explicitly rather than relying on AGENT_START_RESOURCES,
        # so this test stays valid regardless of how the starting economy is tuned.
        agent.resources = {"food": 1, "water": 8, "metal": 1, "energy": 1}
        agent.score = 0.0
        world.end_round()
        # food, metal, energy all set to 1 -> now 0 after depletion -> 3 zero-penalties
        assert agent.score == pytest.approx(3 * PENALTY_ZERO)

    def test_bonus_applied_when_all_resources_above_threshold(self, world):
        agent = world.agents["agent_0"]
        agent.resources = {r: SURVIVAL_THRESHOLD + 5 for r in RESOURCES}
        agent.score = 0.0
        world.end_round()
        assert agent.score == pytest.approx(REWARD_THRESHOLD)

    def test_no_bonus_or_penalty_when_resources_are_mid_range(self, world):
        agent = world.agents["agent_0"]
        # all resources sit strictly between 1 and SURVIVAL_THRESHOLD after depletion
        agent.resources = {r: SURVIVAL_THRESHOLD for r in RESOURCES}
        agent.score = 0.0
        world.end_round()
        # depletes to SURVIVAL_THRESHOLD - 1, none hit zero, not all >= threshold
        assert agent.score == 0.0

    def test_resource_tiles_respawn_after_countdown(self, world):
        tile = ResourceTile(resource_type="food", position=(0, 0), amount=0, respawn_in=1)
        world.resource_tiles = [tile]
        world.end_round()
        assert tile.amount > 0
        assert tile.respawn_in == 0


# ─────────────────────────────────────────────
# Death condition
# ─────────────────────────────────────────────

class TestDeathCondition:
    def test_agent_dies_when_two_resources_hit_zero(self, world):
        agent = world.agents["agent_0"]
        agent.resources = {"food": 5, "water": 1, "metal": 1, "energy": 5}
        world.end_round()
        # water and metal deplete to 0 -> 2 zeros -> death
        assert agent.alive is False

    def test_agent_survives_with_only_one_resource_at_zero(self, world):
        agent = world.agents["agent_0"]
        agent.resources = {"food": 5, "water": 1, "metal": 5, "energy": 5}
        world.end_round()
        # only water hits 0 -> still alive
        assert agent.alive is True

    def test_dead_agents_excluded_from_observations(self, world):
        world.agents["agent_0"].alive = False
        obs = world.get_all_observations()
        assert "agent_0" not in obs
        assert "agent_1" in obs and "agent_2" in obs

    def test_state_summary_alive_count(self, world):
        world.agents["agent_0"].alive = False
        summary = world.get_state_summary()
        assert summary["alive_count"] == 2
