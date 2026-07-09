"""
tests/test_symbol_tracker.py — Emergent Negotiation Arena

Unit tests for core.symbol_tracker:
  - SymbolEntry.stability_score
  - convergence detection
  - vocabulary table output
  - extinction logic

Run:
    pytest tests/test_symbol_tracker.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.symbol_tracker import SymbolTracker, SymbolEntry


# ─────────────────────────────────────────────
# SymbolEntry.stability_score
# ─────────────────────────────────────────────

class TestStabilityScore:
    def test_zero_usage_gives_zero_stability(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        assert entry.stability_score == 0.0

    def test_stability_score_matches_formula(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        entry.usage_count = 5
        entry.success_count = 4
        entry.rejection_count = 1
        entry.adopters = ["agent_0", "agent_1"]

        adopter_bonus = min(1.0, len(set(entry.adopters)) / 3)
        usage_bonus = min(1.0, entry.usage_count / 10)
        success_bonus = entry.success_rate
        expected = round((adopter_bonus + usage_bonus + success_bonus) / 3, 3)

        assert entry.success_rate == 0.8
        assert entry.stability_score == expected

    def test_stability_score_caps_bonuses_at_one(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        entry.usage_count = 50  # usage_bonus would be capped at 1.0
        entry.success_count = 50
        entry.rejection_count = 0
        entry.adopters = ["agent_0", "agent_1", "agent_2", "agent_0"]  # >3 adopters capped at 1.0

        # adopter_bonus=1.0, usage_bonus=1.0, success_bonus=1.0 -> stability = 1.0
        assert entry.stability_score == 1.0

    def test_success_rate_zero_when_never_resolved(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        assert entry.success_rate == 0.0

    def test_dominant_context_returns_most_common(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        entry.resource_contexts = [
            {"offer": {"food": 1}, "request": {"water": 1}, "round": 0},
            {"offer": {"food": 1}, "request": {"water": 1}, "round": 1},
            {"offer": {"metal": 1}, "request": {"energy": 1}, "round": 2},
        ]
        assert entry.dominant_context() == "food→water"

    def test_dominant_context_none_when_no_contexts(self):
        entry = SymbolEntry(symbol="X", first_used_by="agent_0", first_used_round=0)
        assert entry.dominant_context() is None


# ─────────────────────────────────────────────
# Convergence detection
# ─────────────────────────────────────────────

class TestConvergence:
    def test_receiving_a_proposal_does_not_make_target_an_adopter(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        # The target merely RECEIVED the symbol — only the proposer has
        # committed to it, and one adopter is not convergence.
        assert tracker.vocabulary["ZRK"].adopters == ["agent_0"]
        assert tracker.convergence_events == []

    def test_convergence_requires_actual_commitment_by_second_agent(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        assert tracker.convergence_events == []
        # agent_1 accepts a trade under the symbol -> that is adoption
        tracker.record_response_symbol("ZRK", "agent_1", round_number=2)
        assert len(tracker.convergence_events) == 1
        evt = tracker.convergence_events[0]
        assert evt["symbol"] == "ZRK"
        assert evt["context"] == "food→water"
        assert set(evt["adopters"]) == {"agent_0", "agent_1"}

    def test_convergence_via_proposal_reuse_by_second_agent(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        # agent_1 REUSES the symbol in its own proposal -> genuine adoption
        tracker.record_proposal_symbol(
            "ZRK", "agent_1", "agent_2", {"food": 1}, {"water": 1}, round_number=3
        )
        assert len(tracker.convergence_events) == 1
        assert set(tracker.convergence_events[0]["adopters"]) == {"agent_0", "agent_1"}

    def test_no_convergence_with_single_participant(self):
        tracker = SymbolTracker()
        # proposer_id == target_id would be unrealistic, but a symbol used with only
        # one adopter recorded so far should not converge
        entry_symbol = "SOLO"
        tracker.vocabulary[entry_symbol] = SymbolEntry(
            symbol=entry_symbol, first_used_by="agent_0", first_used_round=0
        )
        tracker.vocabulary[entry_symbol].usage_count = 1
        tracker.vocabulary[entry_symbol].adopters = ["agent_0"]
        tracker.vocabulary[entry_symbol].resource_contexts = [
            {"offer": {"food": 1}, "request": {"water": 1}, "round": 0}
        ]
        tracker._check_convergence(entry_symbol, 0)
        assert tracker.convergence_events == []

    def test_response_symbol_unknown_or_blank_is_ignored(self):
        tracker = SymbolTracker()
        tracker.record_response_symbol("NEVER_PROPOSED", "agent_1", round_number=1)
        tracker.record_response_symbol("", "agent_1", round_number=1)
        assert tracker.vocabulary == {}
        assert tracker.convergence_events == []

    def test_duplicate_convergence_not_logged_twice(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        tracker.record_response_symbol("ZRK", "agent_1", round_number=1)
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_2", {"food": 2}, {"water": 2}, round_number=2
        )
        tracker.record_response_symbol("ZRK", "agent_2", round_number=2)
        # same symbol + same dominant context -> only logged once
        assert len(tracker.convergence_events) == 1
        # but adopters set on the vocabulary entry itself keeps growing
        assert set(tracker.vocabulary["ZRK"].adopters) == {"agent_0", "agent_1", "agent_2"}

    def test_different_context_creates_new_convergence_event(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        tracker.record_response_symbol("ZRK", "agent_1", round_number=1)
        assert {e["context"] for e in tracker.convergence_events} == {"food→water"}
        # The symbol drifts to a new dominant context through repeated reuse
        for round_number in (5, 6, 7):
            tracker.record_proposal_symbol(
                "ZRK", "agent_1", "agent_2", {"metal": 1}, {"energy": 1},
                round_number=round_number,
            )
        tracker.record_response_symbol("ZRK", "agent_2", round_number=7)
        contexts = {e["context"] for e in tracker.convergence_events}
        assert contexts == {"food→water", "metal→energy"}


# ─────────────────────────────────────────────
# Vocabulary table output
# ─────────────────────────────────────────────

class TestVocabularyTable:
    def test_empty_tracker_reports_no_stable_vocabulary(self):
        tracker = SymbolTracker()
        table = tracker.get_vocabulary_table()
        assert "No stable vocabulary yet" in table

    def test_table_includes_symbol_after_convergence(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        tracker.record_trade_outcome("ZRK", accepted=True, responder_id="agent_1")
        table = tracker.get_vocabulary_table()
        assert "ZRK" in table
        assert "Symbol" in table and "Stability" in table

    def test_get_stable_vocabulary_filters_by_min_stability(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=1
        )
        # stability is low (single use) -> filtered out at a high threshold
        assert tracker.get_stable_vocabulary(min_stability=0.9) == []
        assert len(tracker.get_stable_vocabulary(min_stability=0.0)) == 1

    def test_vocabulary_growth_data_tracks_rounds(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=0
        )
        tracker.end_round(0)
        tracker.record_proposal_symbol(
            "TRIB", "agent_1", "agent_2", {"water": 1}, {"metal": 1}, round_number=1
        )
        tracker.end_round(1)
        growth = tracker.get_vocabulary_growth_data()
        assert growth == [{"round": 0, "unique_symbols": 1}, {"round": 1, "unique_symbols": 2}]


# ─────────────────────────────────────────────
# Extinction logic
# ─────────────────────────────────────────────

class TestExtinction:
    def test_symbol_marked_extinct_after_ten_rounds_unused(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "OLD", "agent_0", "agent_1", {"food": 1}, {"water": 1}, round_number=0
        )
        tracker.end_round(0)
        tracker.end_round(15)  # gap of 15 rounds >= 10 -> extinct

        assert len(tracker.extinction_events) == 1
        evt = tracker.extinction_events[0]
        assert evt["symbol"] == "OLD"
        assert evt["round_extinct"] == 15
        assert evt["total_uses"] == 1

    def test_symbol_not_extinct_within_window(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "FRESH", "agent_0", "agent_1", {"food": 1}, {"water": 1}, round_number=0
        )
        tracker.end_round(0)
        tracker.end_round(5)  # gap of 5 rounds < 10 -> still alive

        assert tracker.extinction_events == []

    def test_recent_reuse_resets_extinction_window(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "REUSED", "agent_0", "agent_1", {"food": 1}, {"water": 1}, round_number=0
        )
        tracker.end_round(0)
        tracker.record_proposal_symbol(
            "REUSED", "agent_1", "agent_2", {"food": 1}, {"water": 1}, round_number=8
        )
        tracker.end_round(8)
        tracker.end_round(15)  # only 7 rounds since round 8 -> not extinct yet

        assert tracker.extinction_events == []


# ─────────────────────────────────────────────
# record_trade_outcome
# ─────────────────────────────────────────────

class TestTradeOutcome:
    def test_accepted_outcome_increments_success_count(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=0
        )
        tracker.record_trade_outcome("ZRK", accepted=True, responder_id="agent_1")
        assert tracker.vocabulary["ZRK"].success_count == 1
        assert tracker.vocabulary["ZRK"].rejection_count == 0

    def test_rejected_outcome_tracks_rejection_symbol_separately(self):
        tracker = SymbolTracker()
        tracker.record_proposal_symbol(
            "ZRK", "agent_0", "agent_1", {"food": 2}, {"water": 2}, round_number=0
        )
        tracker.record_trade_outcome(
            "ZRK", accepted=False, rejection_symbol="NOPE", responder_id="agent_1"
        )
        assert tracker.vocabulary["ZRK"].rejection_count == 1
        assert "NOPE" in tracker.vocabulary
        assert tracker.vocabulary["NOPE"].rejection_count == 1
        assert "agent_1" in tracker.vocabulary["NOPE"].adopters


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
