"""
symbol_tracker.py — Emergent Negotiation Arena

Tracks the "vocabulary" that agents invent spontaneously.

An agent can include any arbitrary string as a `symbol` field in its
trade proposal or rejection. This module records those strings,
maps them to observed trade contexts, and computes:

  - usage_count: how many times this symbol has been used
  - success_rate: fraction of trades using this symbol that were accepted
  - semantic_cluster: what resource context this symbol was used in most
  - adopters: which agents have used or responded to this symbol

Over many rounds, certain symbols stabilise — agents reuse the same
token for the same resource context. That convergence IS the emergent
language. We visualise it as a vocabulary table and lineage tree.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SymbolEntry:
    symbol: str
    first_used_by: str
    first_used_round: int
    usage_count: int = 0
    success_count: int = 0
    rejection_count: int = 0
    # context: what resources were being traded when this symbol appeared
    resource_contexts: List[Dict] = field(default_factory=list)
    # which agents have used this symbol (as proposer or responder)
    adopters: List[str] = field(default_factory=list)
    # free-form notes the tracker infers about meaning
    inferred_meaning: Optional[str] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.rejection_count
        return round(self.success_count / total, 3) if total > 0 else 0.0

    @property
    def stability_score(self) -> float:
        """
        High stability = used many times, consistently in same resource context,
        adopted by multiple agents.
        Score 0-1.
        """
        if self.usage_count == 0:
            return 0.0
        adopter_bonus = min(1.0, len(set(self.adopters)) / 3)
        usage_bonus = min(1.0, self.usage_count / 10)
        success_bonus = self.success_rate
        return round((adopter_bonus + usage_bonus + success_bonus) / 3, 3)

    def dominant_context(self) -> Optional[str]:
        """What resource combination appears most in this symbol's usage context."""
        if not self.resource_contexts:
            return None
        context_strings = []
        for ctx in self.resource_contexts:
            offer_str = "+".join(sorted(ctx.get("offer", {}).keys()))
            request_str = "+".join(sorted(ctx.get("request", {}).keys()))
            context_strings.append(f"{offer_str}→{request_str}")
        if not context_strings:
            return None
        return max(set(context_strings), key=context_strings.count)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "first_used_by": self.first_used_by,
            "first_used_round": self.first_used_round,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "stability_score": self.stability_score,
            # sorted: set iteration order varies across processes (hash
            # randomization), which would make vocabulary.json non-reproducible
            "adopters": sorted(set(self.adopters)),
            "dominant_context": self.dominant_context(),
            "inferred_meaning": self.inferred_meaning,
        }


class SymbolTracker:
    """
    Central registry for all invented symbols.

    Agents pass in symbols when making or responding to trades.
    The tracker records, analyses, and exposes the vocabulary.

    Also tracks:
      - vocabulary growth over time (new symbols per round)
      - convergence events (when 2+ agents use same symbol for same context)
      - extinction events (symbols that stop being used)
    """

    def __init__(self):
        self.vocabulary: Dict[str, SymbolEntry] = {}
        self.round_vocabulary_counts: List[Tuple[int, int]] = []  # (round, unique_symbols)
        self.convergence_events: List[dict] = []
        self.extinction_events: List[dict] = []
        self._extinct_symbols: set = set()  # symbols already logged as extinct, to avoid re-logging every subsequent round
        self._current_round = 0
        self._round_new_symbols: List[str] = []

    # ── Registration ──

    def record_proposal_symbol(
        self,
        symbol: str,
        proposer_id: str,
        target_id: str,
        offer: Dict[str, int],
        request: Dict[str, int],
        round_number: int,
    ):
        """Called when an agent sends a trade proposal with a symbol."""
        if not symbol or not symbol.strip():
            return

        symbol = symbol.strip()
        context = {"offer": offer, "request": request, "round": round_number}

        if symbol not in self.vocabulary:
            entry = SymbolEntry(
                symbol=symbol,
                first_used_by=proposer_id,
                first_used_round=round_number,
            )
            self.vocabulary[symbol] = entry
            self._round_new_symbols.append(symbol)

        entry = self.vocabulary[symbol]
        entry.usage_count += 1
        entry.resource_contexts.append(context)
        # Adoption means committing to a convention. The proposer commits by
        # using the symbol; the target has merely RECEIVED it at this point and
        # is only counted via record_response_symbol() when it accepts a trade
        # under the symbol (or reuses it in its own proposal later).
        if proposer_id not in entry.adopters:
            entry.adopters.append(proposer_id)
        # If this symbol had previously gone extinct, it's back in use — allow
        # it to be flagged extinct again later if it goes quiet a second time.
        self._extinct_symbols.discard(symbol)

        # Check for convergence (same symbol, same context, multiple adopters)
        self._check_convergence(symbol, round_number)

    def record_response_symbol(self, symbol: str, responder_id: str, round_number: int):
        """
        Called when a responder meaningfully commits to a symbol's convention —
        i.e. it accepted a trade proposed under that symbol. Only then does the
        responder count as an adopter; merely receiving a proposal does not.
        """
        symbol = symbol.strip() if symbol else ""
        if not symbol or symbol not in self.vocabulary:
            return
        entry = self.vocabulary[symbol]
        if responder_id not in entry.adopters:
            entry.adopters.append(responder_id)
        self._check_convergence(symbol, round_number)

    def record_trade_outcome(self, symbol: str, accepted: bool,
                             rejection_symbol: str = "", responder_id: str = ""):
        """Called after a trade resolves to update success/rejection counts."""
        if symbol and symbol.strip() in self.vocabulary:
            entry = self.vocabulary[symbol.strip()]
            if accepted:
                entry.success_count += 1
            else:
                entry.rejection_count += 1

        # If rejector used a different symbol, record that too
        if rejection_symbol and rejection_symbol.strip() and not accepted:
            rsym = rejection_symbol.strip()
            if rsym not in self.vocabulary:
                # Rejection symbols are tracked but marked as rejection-context
                entry = SymbolEntry(
                    symbol=rsym,
                    first_used_by=responder_id,
                    first_used_round=self._current_round,
                )
                self.vocabulary[rsym] = entry
            entry = self.vocabulary[rsym]
            entry.usage_count += 1
            entry.rejection_count += 1
            if responder_id not in entry.adopters:
                entry.adopters.append(responder_id)

    def end_round(self, round_number: int):
        """Called at end of each round. Snapshot vocabulary size, check extinctions."""
        self._current_round = round_number
        self.round_vocabulary_counts.append((round_number, len(self.vocabulary)))
        self._round_new_symbols = []

        # Mark symbols that haven't been used in 10 rounds as extinct
        # (only log each symbol's extinction once, not every round it stays extinct)
        for symbol, entry in self.vocabulary.items():
            if symbol in self._extinct_symbols:
                continue
            if entry.usage_count > 0:
                last_used = max(
                    (ctx["round"] for ctx in entry.resource_contexts),
                    default=entry.first_used_round
                )
                if round_number - last_used >= 10:
                    self.extinction_events.append({
                        "symbol": symbol,
                        "round_extinct": round_number,
                        "total_uses": entry.usage_count,
                        "max_stability": entry.stability_score,
                    })
                    self._extinct_symbols.add(symbol)

    # ── Analysis ──

    def _check_convergence(self, symbol: str, round_number: int):
        """
        Convergence = same symbol COMMITTED TO by 2+ agents for the same
        resource context — through proposal reuse or trade acceptance, never
        through merely receiving a proposal. This is the key signal that a
        'word' has emerged.
        """
        entry = self.vocabulary[symbol]
        unique_adopters = set(entry.adopters)
        dominant = entry.dominant_context()

        if len(unique_adopters) >= 2 and dominant:
            # Check we haven't already logged this convergence
            already_logged = any(
                e["symbol"] == symbol and e["context"] == dominant
                for e in self.convergence_events
            )
            if not already_logged:
                self.convergence_events.append({
                    "symbol": symbol,
                    "round": round_number,
                    "adopters": sorted(unique_adopters),
                    "context": dominant,
                    "stability": entry.stability_score,
                })

    def get_stable_vocabulary(self, min_stability: float = 0.3) -> List[dict]:
        """Returns symbols that have stabilised into 'words'."""
        stable = [
            entry.to_dict()
            for entry in self.vocabulary.values()
            if entry.stability_score >= min_stability
        ]
        return sorted(stable, key=lambda x: x["stability_score"], reverse=True)

    def get_vocabulary_table(self) -> str:
        """Human-readable vocabulary table for demo display."""
        stable = self.get_stable_vocabulary(min_stability=0.1)
        if not stable:
            return "No stable vocabulary yet. Agents are still exploring.\n"

        lines = [
            f"{'Symbol':<20} {'Uses':>5} {'Success%':>9} {'Stability':>10} "
            f"{'Adopters':<20} {'Meaning'}",
            "─" * 90,
        ]
        for entry in stable[:20]:  # top 20
            adopters = ", ".join(entry["adopters"][:3])
            meaning = entry["dominant_context"] or entry["inferred_meaning"] or "unknown"
            lines.append(
                f"{entry['symbol']:<20} {entry['usage_count']:>5} "
                f"{entry['success_rate']*100:>8.1f}% {entry['stability_score']:>10.3f} "
                f"{adopters:<20} {meaning}"
            )
        return "\n".join(lines)

    def get_vocabulary_growth_data(self) -> List[dict]:
        """Returns time-series data for vocabulary growth chart."""
        return [{"round": r, "unique_symbols": n} for r, n in self.round_vocabulary_counts]

    def get_summary(self) -> dict:
        return {
            "total_symbols": len(self.vocabulary),
            "stable_words": len(self.get_stable_vocabulary()),
            "convergence_events": len(self.convergence_events),
            "extinction_events": len(self.extinction_events),
            "most_stable_symbol": max(
                self.vocabulary.values(), key=lambda e: e.stability_score, default=None
            ).symbol if self.vocabulary else None,
            "vocabulary_growth": self.get_vocabulary_growth_data(),
        }

    def to_json(self, path: str = "vocabulary.json"):
        """Save vocabulary to JSON for the HF Space and README artefact."""
        data = {
            "summary": self.get_summary(),
            "vocabulary": [e.to_dict() for e in self.vocabulary.values()],
            "convergence_events": self.convergence_events,
            "extinction_events": self.extinction_events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
