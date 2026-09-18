from src.metrics import (chars_saved_per_second, keystroke_rate,
                         keystrokes_saved, trim_prediction)


class TestKeystrokesSaved:
    def test_counts_correct_leading_run(self):
        assert keystrokes_saved("generate", "gen") == 3

    def test_stops_at_first_wrong_character(self):
        # "ge" matches, "n" != "t" -> the user edits from there
        assert keystrokes_saved("generate", "getting") == 2

    def test_no_overlap(self):
        assert keystrokes_saved("xyz", "gen") == 0

    def test_empty_inputs(self):
        assert keystrokes_saved("", "gen") == 0
        assert keystrokes_saved("gen", "") == 0

    def test_prediction_shorter_than_reference(self):
        assert keystrokes_saved("ge", "generate") == 2

    def test_is_case_sensitive(self):
        # raw characters: the keyboard does not normalise
        assert keystrokes_saved("Gen", "gen") == 0


class TestKeystrokeRate:
    def test_fraction_of_reference(self):
        assert keystroke_rate("gene", "generate") == 0.5

    def test_empty_reference_is_zero_not_error(self):
        assert keystroke_rate("abc", "") == 0.0

    def test_full_match(self):
        assert keystroke_rate("gen", "gen") == 1.0


class TestCharsSavedPerSecond:
    def test_basic_rate(self):
        assert chars_saved_per_second(4, 400) == 10.0

    def test_zero_latency_guarded(self):
        assert chars_saved_per_second(4, 0) == 0.0

    def test_negative_latency_guarded(self):
        assert chars_saved_per_second(4, -1) == 0.0

    def test_none_latency_guarded(self):
        assert chars_saved_per_second(4, None) == 0.0


class TestPhase1Reproduction:
    """Existing behaviour the keystroke work must not disturb."""

    def test_trim_prediction_next_word(self):
        assert trim_prediction("weather today", "weather", "next_word") == "weather"

    def test_trim_prediction_partial_word(self):
        assert trim_prediction("erate", "era", "partial_word") == "era"
