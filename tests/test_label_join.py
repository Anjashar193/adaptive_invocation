"""The join backs every agreement number in the thesis.

If it silently degrades, the reported statistics change without anything
failing - so the expected recovery rate is asserted, not logged.
"""

import csv

import pytest

from src.label_join import (EXPECTED_MATCHES, EXPECTED_TOTAL, assert_join_health,
                            join_labels_to_results, normalize_candidates_from_label,
                            normalize_candidates_from_results)


class TestCandidateParsing:
    def test_splits_on_numbering_not_pipes(self):
        # a literal "|" inside a candidate must NOT split it - this is the bug
        # that dropped the naive implementation to 259/300
        assert normalize_candidates_from_label("1. a | b | 2. c") == ("a | b", "c")

    def test_plain_list(self):
        assert normalize_candidates_from_label("1. foo | 2. bar") == ("foo", "bar")

    def test_collapses_internal_whitespace(self):
        assert normalize_candidates_from_label("1. foo\r\n  bar") == ("foo bar",)

    def test_results_side_splits_on_pipe(self):
        assert normalize_candidates_from_results("foo | bar") == ("foo", "bar")

    def test_empty_parts_dropped(self):
        assert normalize_candidates_from_label("") == ()


class TestRealJoin:
    @pytest.fixture(scope="class")
    @classmethod
    def joined(cls):
        with open("hand_labels.csv", newline="", encoding="utf-8") as f:
            labels = list(csv.DictReader(f))
        with open("phase1_results_v2.csv", newline="", encoding="utf-8") as f:
            results = list(csv.DictReader(f))
        return join_labels_to_results(labels, results)

    def test_recovers_expected_rows(self, joined):
        matched, unmatched = joined
        assert len(matched) == EXPECTED_MATCHES
        assert len(matched) + len(unmatched) == EXPECTED_TOTAL

    def test_only_known_row_is_unmatched(self, joined):
        _, unmatched = joined
        assert {r["id"] for r in unmatched} == {"6"}

    def test_ambiguous_keys_exist_but_never_conflict(self, joined):
        """8 keys map to several result rows; collapsing to the first is only
        safe because they all carry the same usefulness."""
        matched, _ = joined
        assert sum(1 for j in matched if j["match_count"] > 1) == 8
        assert not any(j["usefulness_conflict"] for j in matched)

    def test_health_check_passes(self, joined):
        assert_join_health(*joined)

    def test_health_check_rejects_degraded_join(self, joined):
        matched, unmatched = joined
        with pytest.raises(AssertionError, match="join recovered"):
            assert_join_health(matched[:10], unmatched)
