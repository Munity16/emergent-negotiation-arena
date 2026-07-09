"""
grid_world.py — Emergent Negotiation Arena

A minimal 10x10 grid world with:
  - 3 agents, each with private resource stocks
  - 4 resource types (food, water, metal, energy)
  - Scarcity: each resource spawns on a limited subset of tiles
  - Agents must trade with each other to survive
  - No pre-defined language — agents invent symbols through trial and error

Grid layout:
  F = food zone   (top-left quadrant)
  W = water zone  (top-right quadrant)
  M = metal zone  (bottom-left quadrant)
  E = energy zone (bottom-right quadrant)
  . = empty passable tile
  A = agent position (rendered at runtime)
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

GRID_SIZE = 10
RESOURCES = ["food", "water", "metal", "energy"]


def valid_resource_dict(data) -> bool:
    """
    Strict validation for trade offer/request dicts built from LLM output.

    Prompts ask for positive integers over known resources, but prompts are
    not validators: a negative amount would reverse the transfer direction
    (silently stealing resources) and an unknown resource name would corrupt
    or crash inventory updates. Both must be rejected before any state change.
    """
    if not isinstance(data, dict) or not data:
        return False
    for resource, amount in data.items():
        if resource not in RESOURCES:
            return False
        # bool is a subclass of int — {"food": True} must not pass
        if isinstance(amount, bool) or not isinstance(amount, int):
            return False
        if amount <= 0:
            return False
    return True

# Each agent needs all 4 resources to survive.
# Running to 0 on any resource = penalty; all above threshold = bonus.
SURVIVAL_THRESHOLD = 3
PENALTY_ZERO = -5
REWARD_THRESHOLD = 2
REWARD_TRADE_SUCCESS = 4   # both parties gain this when a trade completes
PENALTY_FAILED_TRADE = -1  # cost of attempting a trade that is rejected

# Resource zones: which (row, col) tiles can spawn each resource
RESOURCE_ZONES = {
    "food":   [(r, c) for r in range(0, 5) for c in range(0, 5)],
    "water":  [(r, c) for r in range(0, 5) for c in range(5, 10)],
    "metal":  [(r, c) for r in range(5, 10) for c in range(0, 5)],
    "energy": [(r, c) for r in range(5, 10) for c in range(5, 10)],
}

# Starting positions for agents (spread across grid)
AGENT_START_POSITIONS = {
    "agent_0": (2, 2),   # near food zone
    "agent_1": (2, 7),   # near water zone
    "agent_2": (7, 5),   # between metal & energy zones
}

# Each agent starts rich in the resource nearest to them, poor in others
AGENT_START_RESOURCES = {
    # Resources deplete by 1/round automatically; death occurs when 2+ resources
    # hit zero simultaneously. The old values (e.g. water=1) meant an agent could
    # die before its trading partner even got a chance to respond to a proposal
    # (a full propose -> accept/reject cycle takes a minimum of 2 rounds). These
    # values give each agent 2-3 rounds of buffer on its scarce resources.
    # agent_2 gets a deeper buffer in its specialties: it is the sole producer
    # of BOTH metal and energy and starts out of visual range of the others,
    # so it needs enough stock to migrate north and still have surplus to sell.
    "agent_0": {"food": 8, "water": 5, "metal": 5, "energy": 5},
    "agent_1": {"food": 5, "water": 8, "metal": 5, "energy": 5},
    "agent_2": {"food": 5, "water": 5, "metal": 7, "energy": 7},
}

# Economy balance: each agent consumes 4 units/round (1 of each resource) but
# can visit only one tile per round, so per-visit collection must exceed 4 or
# the arena is a structural death march no policy can survive (the original
# min(2, amount) yield meant total possible income could never cover total
# consumption — every run ended with all agents dead by round ~8). With
# 12 tiles yielding 3-6 units on a 2-round respawn, world income (~17/round)
# modestly exceeds world consumption (12/round) — survivable for agents that
# farm their zone and trade for the rest, while starvation stays possible.
COLLECT_PER_VISIT = 4
TILE_AMOUNT_MIN = 3
TILE_AMOUNT_MAX = 6
RESPAWN_ROUNDS = 2

# Consumption cadence: agents burn 1 of each resource every DEPLETION_INTERVAL
# rounds (first tick at round 1). At 1/round (the original), an agent with
# far-away resource types needed ~3 units/round from trades while the trade
# loop can move at most ~1 unit/round — mathematically unwinnable, every run
# ended in total death by round 5. Every third round keeps starvation real
# (an idle or isolated agent still dies by ~round 15) while leaving enough
# slack for the travel + trade cycles that keep the three-zone economy alive.
DEPLETION_INTERVAL = 3


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class AgentState:
    agent_id: str
    position: Tuple[int, int]
    resources: Dict[str, int]
    alive: bool = True
    score: float = 0.0
    trades_completed: int = 0
    trades_attempted: int = 0

    def is_desperate(self) -> Dict[str, bool]:
        """Returns which resources the agent critically needs."""
        return {r: self.resources[r] < SURVIVAL_THRESHOLD for r in RESOURCES}

    def surplus(self) -> Dict[str, int]:
        """Returns resources the agent has above survival threshold."""
        return {r: max(0, self.resources[r] - SURVIVAL_THRESHOLD) for r in RESOURCES}

    def to_observation(self) -> dict:
        """Minimal observation dict passed to the LLM agent."""
        return {
            "agent_id": self.agent_id,
            "position": list(self.position),
            "resources": dict(self.resources),
            "desperate_for": [r for r, v in self.is_desperate().items() if v],
            "surplus_of": [r for r, v in self.surplus().items() if v > 0],
            "score": round(self.score, 2),
            "trades_completed": self.trades_completed,
        }


@dataclass
class ResourceTile:
    resource_type: str
    position: Tuple[int, int]
    amount: int = 3
    respawn_in: int = 0  # rounds until this tile respawns


@dataclass
class TradeProposal:
    proposer_id: str
    target_id: str
    offer: Dict[str, int]       # what proposer gives
    request: Dict[str, int]     # what proposer wants back
    symbol: str = ""            # invented symbol the proposer used to describe this trade
    round_number: int = 0


@dataclass
class TradeResult:
    proposal: TradeProposal
    accepted: bool
    rejection_symbol: str = ""  # symbol target used to say "no"


# ─────────────────────────────────────────────
# Grid World
# ─────────────────────────────────────────────

class GridWorld:
    """
    The environment. Manages:
      - Agent positions and resources
      - Resource tile spawning and collection
      - Trade execution and validation
      - Reward calculation
      - Observation generation per agent
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.round: int = 0
        self.agents: Dict[str, AgentState] = {}
        self.resource_tiles: List[ResourceTile] = []
        self.trade_history: List[TradeResult] = []
        self.event_log: List[dict] = []

        self._init_agents()
        self._spawn_resource_tiles()

    # ── Initialisation ──

    def _init_agents(self):
        for agent_id, pos in AGENT_START_POSITIONS.items():
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                position=pos,
                resources=dict(AGENT_START_RESOURCES[agent_id]),
            )

    def _spawn_resource_tiles(self):
        """Place resource tiles: 3 per type in its home zone, plus a commons."""
        self.resource_tiles = []
        for resource, zones in RESOURCE_ZONES.items():
            chosen = random.sample(zones, min(3, len(zones)))
            for pos in chosen:
                self.resource_tiles.append(ResourceTile(
                    resource_type=resource,
                    position=pos,
                    amount=random.randint(TILE_AMOUNT_MIN, TILE_AMOUNT_MAX),
                ))
        # A small central commons — one tile of each resource around the map
        # centre. It softens (without erasing) the quadrant scarcity: an agent
        # CAN top up a foreign resource here, but the tiles are shared and
        # contested, so specialising + trading stays the better strategy. It
        # also gives the three agents a marketplace where their paths cross.
        commons = {"food": (4, 4), "water": (4, 5), "metal": (5, 4), "energy": (5, 5)}
        for resource, pos in commons.items():
            self.resource_tiles.append(ResourceTile(
                resource_type=resource,
                position=pos,
                amount=random.randint(TILE_AMOUNT_MIN, TILE_AMOUNT_MAX),
            ))

    # ── Observation ──

    def get_observation(self, agent_id: str) -> dict:
        """
        Returns everything an agent needs to decide what to do this round.
        Includes nearby resources, nearby agents, and own state.
        """
        agent = self.agents[agent_id]
        ar, ac = agent.position

        # Nearby resource tiles (within 3 tiles)
        nearby_resources = []
        for tile in self.resource_tiles:
            tr, tc = tile.position
            if abs(tr - ar) <= 3 and abs(tc - ac) <= 3 and tile.amount > 0:
                nearby_resources.append({
                    "resource": tile.resource_type,
                    "position": list(tile.position),
                    "amount": tile.amount,
                    "distance": abs(tr - ar) + abs(tc - ac),
                })

        # Visible agents (within 5 tiles)
        visible_agents = []
        for other_id, other in self.agents.items():
            if other_id == agent_id or not other.alive:
                continue
            or_, oc = other.position
            dist = abs(or_ - ar) + abs(oc - ac)
            if dist <= 5:
                desperation = other.is_desperate()
                visible_agents.append({
                    "agent_id": other_id,
                    "position": list(other.position),
                    "distance": dist,
                    # Partial info: agent sees other's desperation signals but not exact counts
                    "appears_desperate": [
                        resource for resource, is_low in desperation.items() if is_low
                    ],
                })

        return {
            "round": self.round,
            "self": agent.to_observation(),
            "nearby_resources": sorted(nearby_resources, key=lambda x: x["distance"]),
            "visible_agents": visible_agents,
            "grid_size": GRID_SIZE,
        }

    def get_all_observations(self) -> Dict[str, dict]:
        return {aid: self.get_observation(aid) for aid in self.agents if self.agents[aid].alive}

    # ── Actions ──

    def move_agent(self, agent_id: str, direction: str) -> bool:
        """Move agent in a cardinal direction. Returns True if move was valid."""
        agent = self.agents[agent_id]
        r, c = agent.position
        deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "stay": (0, 0)}
        dr, dc = deltas.get(direction, (0, 0))
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            agent.position = (nr, nc)
            self._try_collect_resource(agent_id)
            return True
        return False

    def _try_collect_resource(self, agent_id: str):
        """If agent lands on a resource tile, collect it."""
        agent = self.agents[agent_id]
        for tile in self.resource_tiles:
            if tile.position == agent.position and tile.amount > 0:
                collect = min(COLLECT_PER_VISIT, tile.amount)
                agent.resources[tile.resource_type] += collect
                tile.amount -= collect
                if tile.amount == 0:
                    tile.respawn_in = RESPAWN_ROUNDS
                self._log(f"{agent_id} collected {collect} {tile.resource_type} at {tile.position}")

    def execute_trade(self, proposal: TradeProposal, accepted: bool,
                      rejection_symbol: str = "") -> TradeResult:
        """
        Execute a trade between two agents.
        Both must have the resources they're offering.
        """
        result = TradeResult(proposal=proposal, accepted=accepted,
                             rejection_symbol=rejection_symbol)

        proposer = self.agents[proposal.proposer_id]
        target = self.agents[proposal.target_id]

        proposer.trades_attempted += 1

        if accepted:
            # Semantic validation before any transfer: unknown resource names
            # or non-positive amounts must never mutate state.
            if not (valid_resource_dict(proposal.offer)
                    and valid_resource_dict(proposal.request)):
                proposer.score += PENALTY_FAILED_TRADE
                result.accepted = False
                self._log(
                    f"TRADE FAILED (invalid offer/request): "
                    f"{proposal.proposer_id} → {proposal.target_id} "
                    f"| offer {proposal.offer} | request {proposal.request}"
                )
                self.trade_history.append(result)
                return result

            # Validate proposer has enough to offer
            can_offer = all(
                proposer.resources.get(r, 0) >= amt
                for r, amt in proposal.offer.items()
            )
            # Validate target has enough to offer back
            can_return = all(
                target.resources.get(r, 0) >= amt
                for r, amt in proposal.request.items()
            )

            if can_offer and can_return:
                # Transfer resources
                for r, amt in proposal.offer.items():
                    proposer.resources[r] -= amt
                    target.resources[r] += amt
                for r, amt in proposal.request.items():
                    target.resources[r] -= amt
                    proposer.resources[r] += amt

                # Reward both parties
                proposer.score += REWARD_TRADE_SUCCESS
                target.score += REWARD_TRADE_SUCCESS
                proposer.trades_completed += 1
                target.trades_completed += 1
                self._log(
                    f"TRADE SUCCESS: {proposal.proposer_id} ↔ {proposal.target_id} "
                    f"| gave {proposal.offer} | got {proposal.request} "
                    f"| symbol: '{proposal.symbol}'"
                )
            else:
                # Agent lied — penalise
                proposer.score += PENALTY_FAILED_TRADE
                result.accepted = False
                self._log(f"TRADE FAILED (resource mismatch): {proposal.proposer_id} → {proposal.target_id}")
        else:
            proposer.score += PENALTY_FAILED_TRADE
            self._log(
                f"TRADE REJECTED: {proposal.proposer_id} → {proposal.target_id} "
                f"| rejection symbol: '{rejection_symbol}'"
            )

        self.trade_history.append(result)
        return result

    # ── Round advancement ──

    def end_round(self):
        """Called after all agents have acted. Apply survival rewards/penalties and respawn."""
        self.round += 1

        for agent in self.agents.values():
            if not agent.alive:
                continue

            # Survival: consume resources on the depletion cadence
            # (first tick lands on round 1, then every DEPLETION_INTERVAL)
            if self.round % DEPLETION_INTERVAL == 1 or DEPLETION_INTERVAL == 1:
                for r in RESOURCES:
                    agent.resources[r] = max(0, agent.resources[r] - 1)

            # Penalty for any resource hitting zero
            for r in RESOURCES:
                if agent.resources[r] == 0:
                    agent.score += PENALTY_ZERO

            # Bonus for all resources above threshold
            if all(agent.resources[r] >= SURVIVAL_THRESHOLD for r in RESOURCES):
                agent.score += REWARD_THRESHOLD

            # Death check (2 resources at zero = agent dies)
            zeros = sum(1 for r in RESOURCES if agent.resources[r] == 0)
            if zeros >= 2:
                agent.alive = False
                self._log(f"{agent.agent_id} has died (resource depletion)")

        # Respawn resource tiles
        for tile in self.resource_tiles:
            if tile.amount == 0 and tile.respawn_in > 0:
                tile.respawn_in -= 1
                if tile.respawn_in == 0:
                    tile.amount = random.randint(TILE_AMOUNT_MIN, TILE_AMOUNT_MAX)

    # ── Utility ──

    def _log(self, message: str):
        self.event_log.append({"round": self.round, "event": message})

    def get_state_summary(self) -> dict:
        return {
            "round": self.round,
            "agents": {aid: asdict(a) for aid, a in self.agents.items()},
            "alive_count": sum(1 for a in self.agents.values() if a.alive),
            "total_trades": len(self.trade_history),
            "successful_trades": sum(1 for t in self.trade_history if t.accepted),
            "resource_tiles": [
                {"type": t.resource_type, "pos": list(t.position), "amount": t.amount}
                for t in self.resource_tiles
            ],
        }

    def render_ascii(self) -> str:
        """Print ASCII grid for debugging and demo display."""
        zone_map = {pos: res[0].upper() for res, positions in
                    [(r, RESOURCE_ZONES[r]) for r in RESOURCES] for pos in positions}
        agent_positions = {a.position: a.agent_id[-1] for a in self.agents.values() if a.alive}
        resource_positions = {t.position: t.resource_type[0].lower()
                              for t in self.resource_tiles if t.amount > 0}

        lines = [f"  Round {self.round} | " +
                 " | ".join(f"{a.agent_id}: {dict(a.resources)}" for a in self.agents.values())]
        lines.append("  " + "─" * (GRID_SIZE * 2 + 1))
        for r in range(GRID_SIZE):
            row = "  |"
            for c in range(GRID_SIZE):
                pos = (r, c)
                if pos in agent_positions:
                    row += f"\033[1;32m{agent_positions[pos]}\033[0m|"
                elif pos in resource_positions:
                    row += f"\033[0;33m{resource_positions[pos]}\033[0m|"
                elif pos in zone_map:
                    row += f"\033[2m{zone_map[pos]}\033[0m|"
                else:
                    row += ".|"
            lines.append(row)
        lines.append("  " + "─" * (GRID_SIZE * 2 + 1))
        return "\n".join(lines)
