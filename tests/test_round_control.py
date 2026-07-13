"""
UI round-control sync: the Number of Rounds slider must mirror reality.

Replay runs are byte-exact playbacks of a recorded log — the round count is
fixed by the recording, so selecting `replay` locks the slider to that value.
Every other backend re-enables free choice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.replay import load_replay_file

import ui.app as app


class TestUpdateRoundControl:
    def test_replay_locks_slider_to_recorded_round_count(self):
        _, recorded = load_replay_file(app.REPLAY_FILE)
        update = app.update_round_control("replay")
        assert update["value"] == len(recorded)
        assert update["interactive"] is False
        assert "recorded" in update["info"]

    def test_other_backends_re_enable_slider(self):
        for backend in ("auto", "fireworks", "heuristic"):
            update = app.update_round_control(backend)
            assert update["interactive"] is True
            # No forced value: the user keeps their chosen round count.
            assert "value" not in update

    def test_missing_replay_file_leaves_slider_untouched(self, monkeypatch):
        monkeypatch.setattr(app, "REPLAY_FILE", "outputs/does_not_exist.json")
        update = app.update_round_control("replay")
        assert "value" not in update
        assert "interactive" not in update
