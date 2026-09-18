from src.agreement import (agreement_rates, confusion_matrix_3x3, kappa,
                           paired_bootstrap_spearman_delta, spearman, verdict)


class TestKappa:
    def test_perfect_agreement(self):
        scores = [0, 1, 2, 0, 1, 2]
        assert kappa(scores, scores) == 1.0

    def test_linear_weighting_is_more_forgiving(self):
        """Most disagreement here is one point apart, which is why the
        weighted variant is the headline number."""
        human = [0, 1, 2, 0, 1, 2]
        auto = [0, 0, 1, 0, 1, 2]  # two off-by-one errors
        assert kappa(human, auto, weights="linear") > kappa(human, auto)

    def test_degenerate_input_does_not_raise(self):
        assert kappa([1, 1, 1], [1, 1, 1]) == 0.0


class TestAgreementRates:
    def test_counts_and_direction(self):
        human = [2, 2, 1, 0]
        auto = [0, 1, 1, 0]  # two too low, one two-point error
        rates = agreement_rates(human, auto)
        assert rates["n"] == 4
        assert rates["exact"] == 0.5
        assert rates["off_by_one"] == 0.25
        assert rates["off_by_two"] == 0.25
        assert rates["auto_too_low"] == 2
        assert rates["auto_too_high"] == 0

    def test_empty(self):
        assert agreement_rates([], []) == {}


class TestConfusionMatrix:
    def test_all_cells_present(self):
        counts = confusion_matrix_3x3([0, 1], [1, 1])
        assert len(counts) == 9
        assert counts[(0, 1)] == 1
        assert counts[(1, 1)] == 1
        assert counts[(2, 2)] == 0


class TestSpearman:
    def test_monotone_relationship(self):
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_constant_input_is_zero_not_nan(self):
        assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


class TestPairedBootstrap:
    def test_is_deterministic_under_fixed_seed(self):
        human = [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
        good = [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
        bad = [2, 1, 0, 2, 1, 0, 1, 0, 2, 1]
        first = paired_bootstrap_spearman_delta(human, good, bad, n_boot=200, seed=7)
        second = paired_bootstrap_spearman_delta(human, good, bad, n_boot=200, seed=7)
        assert first == second

    def test_detects_the_better_metric(self):
        human = [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
        good = [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
        bad = [2, 1, 0, 2, 1, 0, 1, 0, 2, 1]
        ci = paired_bootstrap_spearman_delta(human, good, bad, n_boot=500, seed=1)
        assert ci["delta"] > 0
        assert verdict(ci) == "a_better"

    def test_identical_metrics_are_a_tie(self):
        human = [0, 1, 2, 0, 1, 2, 1, 2, 0, 1]
        metric = [0, 1, 1, 0, 2, 2, 1, 2, 0, 0]
        ci = paired_bootstrap_spearman_delta(human, metric, metric, n_boot=300, seed=3)
        assert ci["delta"] == 0.0
        assert verdict(ci) == "tie"
