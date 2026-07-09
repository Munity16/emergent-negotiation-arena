"""
llm_agent.py — Emergent Negotiation Arena

LLM-powered agent that:
  1. Reads its observation from the grid world
  2. Decides: move, collect, or attempt a trade
  3. If trading: generates a TradeProposal with an INVENTED SYMBOL
  4. If responding: accepts or rejects with its own symbol

The key constraint that produces emergent language:
  - The agent is NOT told what symbols mean.
  - It invents symbols freely (any string).
  - It learns from whether trades using certain symbols succeed or fail.
  - Over many rounds, shared symbols emerge for shared resource contexts.

LLM backend: Fireworks AI API — set FIREWORKS_API_KEY (env var, never
hardcoded) and AGENT_BACKEND=fireworks. Without a key, the deterministic
heuristic backend and replay mode keep everything runnable.
"""

import os
import json
import re
import asyncio
import httpx
from typing import Optional, Dict, List, Tuple
from core.grid_world import TradeProposal, AgentState, RESOURCES, valid_resource_dict


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
# Defaults only — the live values are read from the environment INSIDE
# LLMClient.__init__ (instance-owned), so main.py's --backend flag and the
# Gradio UI's selectors keep working even though they set env vars after
# this module has been imported.

# Verified live against this key's serverless catalog: fast, cheap, strong
# JSON compliance. Override with FIREWORKS_MODEL.
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/gpt-oss-120b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# "fireworks" calls an LLM; "heuristic" is the deterministic rule-based
# no-key demo backend (see NegotiationAgent._heuristic_action).
# "replay" never reaches this module — it is handled by core/replay.py.
LLM_BACKENDS = ("fireworks",)
KNOWN_BACKENDS = ("fireworks", "heuristic")

MAX_TOKENS = 1500  # Qwen3 is a "thinking" model; its internal reasoning eats the token
                   # budget before reaching the JSON answer, so this needs headroom.
TEMPERATURE = 0.7

# Transient failures worth riding through in a many-round simulation.
# 4xx (auth, bad request) and JSON/validation errors fail immediately.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 0.5


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous agent in a resource-scarcity grid world.
You must survive by trading resources with other agents.

CRITICAL RULES:
1. You have NO shared language with other agents. You must INVENT symbols to communicate.
2. A symbol is any short string you choose: "ZRK", "@@food", "triangle", "GIVE_W", etc.
3. Other agents do not know what your symbols mean. They will learn from context.
4. You learn what their symbols mean by observing which trades succeed or fail.
5. Efficient symbol reuse = other agents learn your vocabulary faster = more trades succeed.

RESOURCES: food, water, metal, energy
You need ALL 4 to survive. You consume 1 of each per round.
Running out of 2+ resources = death.

OUTPUT FORMAT: Respond ONLY with valid JSON. No prose, no explanation."""

PENDING_PROPOSAL_BLOCK = """
==============================
YOU HAVE A PENDING OFFER — RESPOND NOW
{pending_proposal}
You MUST choose ACCEPT_TRADE or REJECT_TRADE this turn. Do not choose MOVE or
PROPOSE_TRADE while an offer is waiting on you — the offer expires if ignored
and the sender will stop trusting your symbols.
==============================
"""

DECISION_PROMPT_TEMPLATE = """{pending_section}Your current state:
{observation}

Symbol history (what has worked before):
{symbol_history}

Choose ONE action:

A) MOVE — move to collect resources
B) PROPOSE_TRADE — propose a trade to another agent  
C) ACCEPT_TRADE — accept the pending proposal (REQUIRED if one is pending)
D) REJECT_TRADE — reject the pending proposal (only other valid choice if one is pending)

Respond with JSON in EXACTLY one of these formats:

For MOVE:
{{"action": "move", "direction": "up|down|left|right|stay", "reasoning": "brief"}}

For PROPOSE_TRADE:
{{"action": "propose_trade", "target_id": "agent_X", "offer": {{"resource": amount}}, "request": {{"resource": amount}}, "symbol": "YOUR_INVENTED_SYMBOL", "reasoning": "brief"}}

For ACCEPT_TRADE:
{{"action": "accept_trade", "symbol": "YOUR_RESPONSE_SYMBOL", "reasoning": "brief"}}

For REJECT_TRADE:
{{"action": "reject_trade", "symbol": "YOUR_REJECTION_SYMBOL", "reasoning": "brief"}}

IMPORTANT: 
- symbol must be a short invented string (2-10 chars), NOT a resource name.
- offer/request values must be positive integers.
- Only propose trades when you have surplus AND another agent is visible AND desperate."""


# ─────────────────────────────────────────────
# Backend health / resolution
# ─────────────────────────────────────────────

class BackendUnavailable(RuntimeError):
    """An explicitly requested backend cannot run. Message says why and what to do."""


def check_fireworks_ready() -> Tuple[bool, str]:
    """Fireworks preflight = key presence. Returns (ready, detail)."""
    if os.getenv("FIREWORKS_API_KEY", ""):
        return True, "FIREWORKS_API_KEY is set"
    return False, "FIREWORKS_API_KEY is not set"


def resolve_backend(requested: str, replay_file: Optional[str] = None) -> Tuple[str, str]:
    """
    Map a requested backend (possibly "auto") to a runnable one.

    Explicit choices are honoured but preflight-checked: an unusable explicit
    backend raises BackendUnavailable with a clear reason instead of producing
    a dead simulation. "auto" falls through fireworks → replay → heuristic so
    a no-key machine still gets a working demo.

    Returns (backend, human_readable_reason).
    """
    if requested == "fireworks":
        ok, detail = check_fireworks_ready()
        if not ok:
            raise BackendUnavailable(
                "FIREWORKS_API_KEY is not set. Export the key, or use "
                "--backend auto / heuristic until it arrives."
            )
        return "fireworks", detail
    if requested == "heuristic":
        return "heuristic", "rule-based agents — deterministic, no LLM or API key needed"
    if requested == "replay":
        if replay_file and os.path.exists(replay_file):
            return "replay", f"replaying recorded run: {replay_file}"
        raise BackendUnavailable(
            f"Replay file not found: {replay_file!r}. Run a simulation first "
            "(e.g. --backend heuristic) or point --replay-file at a saved simulation_log.json."
        )
    if requested == "auto":
        fw_ok, _ = check_fireworks_ready()
        if fw_ok:
            return "fireworks", "auto-selected: FIREWORKS_API_KEY present"
        if replay_file and os.path.exists(replay_file):
            return "replay", f"auto-selected: no live backend, replaying {replay_file}"
        return "heuristic", "auto-selected: no live backend available — rule-based demo mode"
    raise BackendUnavailable(f"Unknown backend {requested!r}. "
                             f"Valid: auto, fireworks, heuristic, replay.")


# ─────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────

class LLMClient:
    """Handles calls to the Fireworks AI API (LLM backend)."""

    def __init__(self):
        # All runtime config is read fresh at instantiation, not from
        # module-level constants frozen at import time: main.py and the Gradio
        # UI set these env vars AFTER this module is imported, so frozen
        # constants would silently ignore the user's backend/model choice.
        self.backend = os.getenv("AGENT_BACKEND", "heuristic")
        self.fireworks_model = os.getenv("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
        self.fireworks_api_key = os.getenv("FIREWORKS_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=90.0)

        if self.backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"Unknown AGENT_BACKEND {self.backend!r}. Valid: {', '.join(KNOWN_BACKENDS)}."
            )
        # Fail fast and clearly here rather than mid-simulation with an
        # opaque 401 from the API.
        if self.backend == "fireworks" and not self.fireworks_api_key:
            raise RuntimeError(
                "AGENT_BACKEND=fireworks but FIREWORKS_API_KEY is not set. "
                "Export the key, or use --backend heuristic / auto until it arrives."
            )

    async def complete(self, system: str, user: str) -> str:
        """Returns the raw text response from the LLM."""
        if self.backend == "heuristic":
            # NegotiationAgent short-circuits before reaching the LLM in
            # heuristic mode; landing here means a plumbing bug.
            raise RuntimeError("heuristic backend must not call the LLM")
        return await self._fireworks_complete(system, user)

    async def _post_json(self, url: str, payload: dict,
                         headers: Optional[dict] = None) -> dict:
        """
        POST with short exponential backoff on transient failures only
        (connection errors, timeouts, 429, 5xx). Auth/validation errors
        surface immediately — retrying them just burns time.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self.client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in RETRYABLE_STATUS:
                    raise
                last_exc = e
            except httpx.TransportError as e:  # covers timeouts + network errors
                last_exc = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
        raise last_exc

    async def _fireworks_complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.fireworks_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            # Fireworks' mechanism for capping the model's reasoning pass.
            # 'low' is the minimum gpt-oss models accept ('none' is rejected
            # with an invalid_request_error); without a cap, reasoning eats
            # MAX_TOKENS before the JSON answer.
            "reasoning_effort": "low",
        }
        data = await self._post_json(
            f"{FIREWORKS_BASE_URL}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.fireworks_api_key}"},
        )
        return data["choices"][0]["message"]["content"]

    async def close(self):
        await self.client.aclose()


# ─────────────────────────────────────────────
# Agent Decision Engine
# ─────────────────────────────────────────────

class NegotiationAgent:
    """
    Wraps an LLM call into a structured agent decision.

    Each agent instance:
      - Has its own symbol memory (what it has used and what worked)
      - Calls the shared LLMClient (one client, 3 agents share it)
      - Returns structured AgentAction objects
    """

    def __init__(self, agent_id: str, llm_client: LLMClient):
        self.agent_id = agent_id
        self.llm = llm_client
        # Local symbol memory: symbol → {"uses": int, "successes": int, "context": str}
        self.symbol_memory: Dict[str, dict] = {}
        # "visible" (demo): the responder sees the full offer/request payload.
        # "hidden" (research): the responder sees only the symbol plus coarse
        # hints, so accepting requires actually inferring the symbol's meaning
        # from its own trade history — the stronger emergent-communication
        # setting. Read at instantiation for the same reason as AGENT_BACKEND.
        self.semantics_mode = os.getenv("SEMANTICS_MODE", "visible")

    def _build_symbol_history(self) -> str:
        """Compact summary of this agent's own symbol experiences."""
        if not self.symbol_memory:
            return "None yet. Invent your first symbol."
        lines = []
        for sym, stats in list(self.symbol_memory.items())[-10:]:  # last 10
            rate = stats["successes"] / max(1, stats["uses"])
            lines.append(
                f"  '{sym}': used {stats['uses']}x, "
                f"success {rate:.0%}, context: {stats.get('context', '?')}"
            )
        return "\n".join(lines)

    def update_symbol_memory(self, symbol: str, success: bool, context: str):
        """Called after a trade resolves."""
        if not symbol:
            return
        if symbol not in self.symbol_memory:
            self.symbol_memory[symbol] = {"uses": 0, "successes": 0, "context": context}
        self.symbol_memory[symbol]["uses"] += 1
        if success:
            self.symbol_memory[symbol]["successes"] += 1

    async def decide(
        self,
        observation: dict,
        pending_proposal: Optional[TradeProposal] = None,
    ) -> dict:
        """
        Main decision loop. Returns a dict with action and parameters.
        Falls back to a safe default if LLM fails or returns invalid JSON.
        """
        if self.llm.backend == "heuristic":
            return self._heuristic_action(observation, pending_proposal)

        pending_section = ""
        if pending_proposal:
            if self.semantics_mode == "hidden":
                # Research mode: the symbol IS the message. The responder gets
                # no offer/request semantics — only its own past experience
                # with this symbol and a coarse size hint. The world still
                # keeps the real offer/request internally for execution.
                pending_payload = {
                    "from": pending_proposal.proposer_id,
                    "symbol": pending_proposal.symbol,
                    "offer_size_hint": sum(pending_proposal.offer.values()),
                    "your_past_experience_with_this_symbol": self.symbol_memory.get(
                        pending_proposal.symbol,
                        "never seen before — accept or reject on risk tolerance",
                    ),
                }
            else:
                pending_payload = {
                    "from": pending_proposal.proposer_id,
                    "offer": pending_proposal.offer,
                    "request": pending_proposal.request,
                    "symbol": pending_proposal.symbol,
                }
            pending_str = json.dumps(pending_payload, indent=2)
            pending_section = PENDING_PROPOSAL_BLOCK.format(pending_proposal=pending_str)

        user_prompt = DECISION_PROMPT_TEMPLATE.format(
            pending_section=pending_section,
            observation=json.dumps(observation, indent=2),
            symbol_history=self._build_symbol_history(),
        )

        try:
            raw = await self.llm.complete(SYSTEM_PROMPT, user_prompt)
            action = self._parse_response(raw)
            return action
        except Exception as e:
            print(f"[{self.agent_id}] LLM error: {e} — using fallback")
            return self._fallback_action(observation, pending_proposal)

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from LLM response. Handle markdown fences and Qwen3 thinking blocks."""
        # Strip any <think>...</think> block (Qwen3 thinking mode), including an
        # unterminated one that got truncated by max_tokens (no closing tag yet).
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^.*?</think>", "", raw, flags=re.DOTALL)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Find the first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response (len={len(raw)}): {raw[:150]}")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed/truncated JSON ({e}): {match.group()[:150]}")

        # Validate required fields per action type
        action = data.get("action", "")
        if action == "move":
            data.setdefault("direction", "stay")
        elif action == "propose_trade":
            if not data.get("target_id") or not data.get("offer") or not data.get("request"):
                raise ValueError("propose_trade missing required fields")
            # Same strict validator the simulation uses — catch garbage at the
            # parse boundary so the agent falls back instead of emitting a
            # proposal the world would reject anyway.
            if not (valid_resource_dict(data["offer"]) and valid_resource_dict(data["request"])):
                raise ValueError(
                    f"propose_trade failed resource validation: "
                    f"offer={data.get('offer')!r} request={data.get('request')!r}"
                )
            data.setdefault("symbol", f"SYM{len(self.symbol_memory)}")
        elif action in ("accept_trade", "reject_trade"):
            data.setdefault("symbol", f"RSP{len(self.symbol_memory)}")
        else:
            raise ValueError(f"Unknown action: {action}")

        return data

    def _fallback_action(
        self,
        observation: dict,
        pending_proposal: Optional[TradeProposal],
    ) -> dict:
        """
        Safe fallback when LLM fails.
        - If there's a pending proposal and we're desperate for what they offer, accept.
        - Otherwise, move toward nearest needed resource.
        """
        self_obs = observation.get("self", {})
        desperate = self_obs.get("desperate_for", [])
        resources = self_obs.get("resources", {})

        # Accept trade if it gives us something we need
        if pending_proposal:
            offered = list(pending_proposal.offer.keys())
            if any(r in desperate for r in offered):
                return {"action": "accept_trade", "symbol": "OK", "reasoning": "fallback-accept"}
            return {"action": "reject_trade", "symbol": "NO", "reasoning": "fallback-reject"}

        # Move toward nearest needed resource
        nearby = observation.get("nearby_resources", [])
        if nearby and desperate:
            needed = [t for t in nearby if t["resource"] in desperate]
            if needed:
                target = needed[0]
                tr, tc = target["position"]
                ar, ac = self_obs.get("position", [0, 0])
                if tr > ar:
                    direction = "down"
                elif tr < ar:
                    direction = "up"
                elif tc > ac:
                    direction = "right"
                else:
                    direction = "left"
                return {"action": "move", "direction": direction, "reasoning": "fallback-move"}

        return {"action": "move", "direction": "stay", "reasoning": "fallback-stay"}

    # ── Heuristic backend (no-key demo mode) ──

    # Where each resource spawns; used to explore when nothing is in sight.
    _ZONE_CENTERS = {"food": (2, 2), "water": (2, 7), "metal": (7, 2), "energy": (7, 7)}
    # Each agent's home base and the specialties it farms and sells there
    # (mirrors AGENT_START_POSITIONS/RESOURCES). agent_2's home sits between
    # the two southern zones — it is the sole producer of BOTH metal and
    # energy, and from (7, 5) tiles of both are usually in collection range.
    _HOME = {
        "agent_0": ((2, 2), ("food",)),
        "agent_1": ((2, 7), ("water",)),
        "agent_2": ((7, 5), ("metal", "energy")),
    }
    # A traveling seller wants this much of each specialty before leaving home.
    _CARGO_TARGET = 6
    _HEUR_ACCEPT_SYMBOL = "ACK"
    _HEUR_REJECT_SYMBOL = "NAK"

    def _heuristic_action(
        self,
        observation: dict,
        pending_proposal: Optional[TradeProposal],
    ) -> dict:
        """
        Rule-based decision policy for the heuristic backend.

        Deliberately deterministic — no randomness — for two reasons: replays
        of heuristic runs must reproduce bit-exactly, and agent decisions must
        not consume the world's seeded RNG stream.

        Priorities:
          1. Respond to a pending offer: accept when it supplies a resource we
             aren't rich in and paying the request leaves us at least 1 of
             each requested resource.
          2. Harvest the tile underfoot.
          3. Propose to a visible partner for a needed resource that CANNOT be
             farmed nearby (trade is the only channel for remote resources —
             farming metal to a mountain while water sits at zero is how
             agents die rich). Request something the target does NOT look
             short on; offer a resource we can spare, preferably one the
             target is short on. Reuse whichever symbol already succeeded for
             that offer→request context (this reuse is what lets a convention
             spread between agents).
          4. Walk to a nearby tile of a resource we need.
          5. Go home and restock when specialty cargo runs out.
          6. Stock up on specialty cargo before leaving home to sell.
          7. When isolated, migrate toward the NEAREST zone of an unmeetable
             need — its producer is also a future trading partner. While
             partners are visible, stay put and keep trading instead.
          8. Otherwise top up from any nearby tile, or explore toward the
             scarcest resource's zone.
        """
        self_obs = observation.get("self", {})
        resources = dict(self_obs.get("resources", {}))

        # 1. Respond to a pending offer
        if pending_proposal:
            offered = pending_proposal.offer
            requested = pending_proposal.request
            # Generous acceptance: with slow depletion, stockpiling a not-yet-
            # scarce resource is still worth it, and accepted trades are what
            # make symbols converge.
            gives_useful = any(resources.get(r, 0) <= 6 for r in offered)
            can_afford = all(resources.get(r, 0) - amt >= 1 for r, amt in requested.items())
            if gives_useful and can_afford:
                return {"action": "accept_trade", "symbol": self._HEUR_ACCEPT_SYMBOL,
                        "reasoning": "heuristic-accept"}
            return {"action": "reject_trade", "symbol": self._HEUR_REJECT_SYMBOL,
                    "reasoning": "heuristic-reject"}

        pos = list(self_obs.get("position", [0, 0]))
        nearby = [t for t in observation.get("nearby_resources", []) if t["amount"] > 0]
        nearby_types = {t["resource"] for t in nearby}
        visible = observation.get("visible_agents", [])
        # Everything we're running low on, most critical first; ties broken by
        # resource name so the policy stays deterministic.
        needs = sorted(
            (r for r in RESOURCES if resources.get(r, 0) <= 4),
            key=lambda r: (resources.get(r, 0), r),
        )
        unmeetable = [r for r in needs if r not in nearby_types]
        critical = [r for r in unmeetable if resources.get(r, 0) <= 1]

        # 2a. Life-or-death trades outrank farming: a stock at ≤1 with no
        # local source must be bought NOW, not after topping up the granary.
        if critical:
            proposal = self._propose_for(critical, visible, resources)
            if proposal:
                return proposal

        # 2b. Harvest the tile underfoot — unless that stock is already
        # hoard-level (camping a respawning tile while other stocks die is
        # how an agent starves at 11 water).
        underfoot = [t for t in nearby if t["position"] == pos]
        if underfoot and resources.get(underfoot[0]["resource"], 0) < 9:
            return {"action": "move", "direction": "stay",
                    "reasoning": "heuristic-collect"}

        # 3. Trade for anything else that cannot be farmed here
        proposal = self._propose_for(unmeetable, visible, resources)
        if proposal:
            return proposal

        # 4. Farm a nearby tile of something we need
        needed_tiles = [t for t in nearby if t["resource"] in needs]
        if needed_tiles:
            best = sorted(
                needed_tiles,
                key=lambda t: (resources.get(t["resource"], 0), t["distance"], t["resource"]),
            )[0]
            return {"action": "move", "direction": self._step_toward(pos, best["position"]),
                    "reasoning": "heuristic-seek"}

        home_pos, specialties = self._HOME.get(self.agent_id, (None, ()))

        # 5. A seller whose cargo ran out goes home to farm it (this is what
        # brings the traveling metal/energy producer back south). Specialists
        # already standing in their home zone just farm via rules 2/4.
        if home_pos and any(resources.get(s, 0) <= 4 and s not in nearby_types
                            for s in specialties):
            return {"action": "move",
                    "direction": self._step_toward(pos, home_pos),
                    "reasoning": "heuristic-restock"}

        # 6. Stock up before leaving home: an isolated producer with thin
        # cargo farms its specialties first — arriving at the market with
        # nothing to sell helps nobody.
        if not visible and specialties:
            thin = [s for s in specialties
                    if resources.get(s, 0) < self._CARGO_TARGET]
            cargo_tiles = [t for t in nearby if t["resource"] in thin]
            if cargo_tiles:
                best = sorted(
                    cargo_tiles,
                    key=lambda t: (resources.get(t["resource"], 0), t["distance"], t["resource"]),
                )[0]
                return {"action": "move",
                        "direction": self._step_toward(pos, best["position"]),
                        "reasoning": "heuristic-stockup"}

        # 7. Migrate toward the NEAREST zone that fixes an unmeetable need —
        # walking to adjacent food beats a cross-map march for energy, and
        # the producer living there is a future trading partner. While
        # partners are visible, STAY and keep the local trade economy alive —
        # UNLESS a nearly-dead stock (≤2) is one no visible partner can sell
        # either (all short on it, none produces it): then only its home zone
        # can save us, so march there even if it means leaving the market.
        def zone_distance(r):
            zr, zc = self._ZONE_CENTERS[r]
            return abs(zr - pos[0]) + abs(zc - pos[1])

        if unmeetable and not visible:
            target_resource = min(unmeetable,
                                  key=lambda r: (resources.get(r, 0) + zone_distance(r), r))
            return {"action": "move",
                    "direction": self._step_toward(pos, self._ZONE_CENTERS[target_resource]),
                    "reasoning": "heuristic-migrate"}
        stranded = [
            r for r in unmeetable
            if resources.get(r, 0) <= 2 and all(
                r in v.get("appears_desperate", [])
                and r not in self._HOME.get(v["agent_id"], (None, ()))[1]
                for v in visible
            )
        ]
        if stranded:
            target_resource = min(stranded,
                                  key=lambda r: (resources.get(r, 0) + zone_distance(r), r))
            return {"action": "move",
                    "direction": self._step_toward(pos, self._ZONE_CENTERS[target_resource]),
                    "reasoning": "heuristic-migrate"}

        # 8. Top up from whatever is nearby, or explore
        if nearby:
            best = sorted(
                nearby,
                key=lambda t: (resources.get(t["resource"], 0), t["distance"], t["resource"]),
            )[0]
            return {"action": "move", "direction": self._step_toward(pos, best["position"]),
                    "reasoning": "heuristic-topup"}
        scarcest = min(RESOURCES, key=lambda r: (resources.get(r, 0), r))
        return {"action": "move",
                "direction": self._step_toward(pos, self._ZONE_CENTERS[scarcest]),
                "reasoning": "heuristic-explore"}

    def _propose_for(self, requests: List[str], visible: List[dict],
                     resources: Dict[str, int]) -> Optional[dict]:
        """
        Try to build a trade proposal for one of `requests` (most critical
        first) against the visible partners. Returns None when no partner can
        plausibly pay or we have nothing spare to offer.
        """
        if not requests or not visible:
            return None
        for target in sorted(visible, key=lambda v: (v["distance"], v["agent_id"])):
            target_short_on = set(target.get("appears_desperate", []))
            target_specialties = self._HOME.get(target["agent_id"], (None, ()))[1]
            # Request something the target does NOT look short on — asking a
            # starving agent for its last units is how every proposal ends up
            # rejected. Exception: a producer's own specialty is always worth
            # requesting (they can re-farm it), which keeps late-game trade
            # from deadlocking once everyone looks poor.
            viable = [r for r in requests
                      if r not in target_short_on or r in target_specialties]
            if not viable:
                continue
            request_resource = viable[0]
            # Sell only from a real buffer (≥5): a specialist that trades
            # itself down to the survival threshold starves next tick.
            spare = [r for r in RESOURCES
                     if r != request_resource and resources.get(r, 0) >= 5]
            if not spare:
                return None  # nothing to offer anyone this round
            # Prefer offering what the target is short on, then abundance.
            spare.sort(key=lambda r: (r not in target_short_on,
                                      -resources.get(r, 0), r))
            offer_resource = spare[0]
            offer_amt = 2 if resources.get(offer_resource, 0) >= 7 else 1
            return {
                "action": "propose_trade",
                "target_id": target["agent_id"],
                "offer": {offer_resource: offer_amt},
                "request": {request_resource: 2},
                "symbol": self._heuristic_symbol(offer_resource, request_resource),
                "reasoning": "heuristic-propose",
            }
        return None

    def _heuristic_symbol(self, offer_resource: str, request_resource: str) -> str:
        """
        Reuse the most successful symbol already learned for this exact
        offer→request context; otherwise mint a deterministic new one. The
        context string matches what notify_trade_result stores.
        """
        context = f"['{offer_resource}']→['{request_resource}']"
        candidates = [
            (stats.get("successes", 0), sym)
            for sym, stats in self.symbol_memory.items()
            if stats.get("context") == context and stats.get("successes", 0) > 0
        ]
        if candidates:
            return max(candidates)[1]
        return f"{offer_resource[0].upper()}{request_resource[0].upper()}{self.agent_id[-1]}"

    @staticmethod
    def _step_toward(pos, target) -> str:
        r, c = pos
        tr, tc = target
        if tr > r:
            return "down"
        if tr < r:
            return "up"
        if tc > c:
            return "right"
        if tc < c:
            return "left"
        return "stay"


# ─────────────────────────────────────────────
# Agent Pool — manages all 3 agents in parallel
# ─────────────────────────────────────────────

class AgentPool:
    """
    Manages all 3 NegotiationAgents.
    Key: all agents call the LLM in PARALLEL using asyncio.gather() —
    one round costs one API round-trip instead of three sequential ones.
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.agents: Dict[str, NegotiationAgent] = {
            agent_id: NegotiationAgent(agent_id, self.llm_client)
            for agent_id in ["agent_0", "agent_1", "agent_2"]
        }

    @property
    def backend_name(self) -> str:
        return self.llm_client.backend

    async def decide_all_parallel(
        self,
        observations: Dict[str, dict],
        pending_proposals: Optional[Dict[str, TradeProposal]] = None,
    ) -> Dict[str, dict]:
        """
        Run all 3 agents' decision steps IN PARALLEL.
        Returns {agent_id: action_dict} for each alive agent.
        """
        if pending_proposals is None:
            pending_proposals = {}

        tasks = {}
        for agent_id, obs in observations.items():
            pending = pending_proposals.get(agent_id)
            tasks[agent_id] = self.agents[agent_id].decide(obs, pending)

        # Parallel execution — all 3 LLM calls fire simultaneously
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        decisions = {}
        for agent_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                print(f"[{agent_id}] Decision error: {result}")
                decisions[agent_id] = {"action": "move", "direction": "stay",
                                       "reasoning": "error-fallback"}
            else:
                decisions[agent_id] = result

        return decisions

    def notify_trade_result(self, proposer_id: str, target_id: str,
                            symbol: str, accepted: bool, context: str):
        """Update both agents' symbol memories after a trade resolves."""
        self.agents[proposer_id].update_symbol_memory(symbol, accepted, context)
        self.agents[target_id].update_symbol_memory(symbol, accepted, context)

    async def close(self):
        await self.llm_client.close()
