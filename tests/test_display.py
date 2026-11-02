"""The panel is what the README's GIF depicts, so its geometry is pinned."""

import pytest

from simulst.display import Panel, wrap


class TestWrap:
    def test_short_text_is_one_line(self):
        assert wrap("the meeting was moved", 40) == ["the meeting was moved"]

    def test_long_text_breaks_on_words(self):
        lines = wrap("one two three four five six", 12)
        assert all(len(line) <= 12 for line in lines)
        assert " ".join(lines) == "one two three four five six"

    def test_empty_text_still_yields_a_line(self):
        assert wrap("", 40) == [""]

    def test_a_word_longer_than_the_width_is_not_dropped(self):
        assert "".join(wrap("supercalifragilistic", 8)) == "supercalifragilistic"


class TestPanel:
    def test_height_follows_the_body(self):
        assert Panel("t", "s", body_lines=5).height == 12
        assert Panel("t", "s", body_lines=8).height == 15

    def test_redrawing_is_skipped_without_a_terminal(self, capsys):
        """Piped output must stay append-only or the escape codes end up in logs."""
        panel = Panel("t", "s")
        panel.ansi = False
        panel.update("status", "hello")
        assert capsys.readouterr().out == ""

    def test_close_without_ansi_prints_both_rows(self, capsys):
        panel = Panel("t", "s")
        panel.ansi = False
        panel.close("streamed text", "offline text")
        out = capsys.readouterr().out
        assert "streamed text" in out and "offline text" in out

    def test_the_bar_is_clamped(self):
        panel = Panel("t", "s")
        for fraction in (-1.0, 0.0, 0.5, 1.0, 2.0):
            assert isinstance(panel._bar(fraction), str)

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    def test_the_bar_keeps_a_constant_cell_count(self, fraction):
        panel = Panel("t", "s")
        bar = panel._bar(fraction)
        assert bar.count("\u2501") == panel.width - 22
