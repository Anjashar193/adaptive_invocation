"""Agreement statistics between the automatic scorer and hand labels.

Replaces the ad-hoc analysis scripts that produced the numbers in
PHASE2_METRIC_PLAN.md. Every statistic here is seeded and reproducible.

Two choices worth knowing about:

* **Linear-weighted kappa is the headline number.** 36.8% of disagreements are
  one point apart vs 4.4% two points apart, so treating "plausible vs relevant"
  as badly as "irrelevant vs relevant" understates the scorer. Unweighted kappa
  is reported alongside for comparability.
* **The bootstrap is paired.** Item indices are resampled once per replicate
  and both metrics are scored on that same resample. Resampling the two metrics
  independently would discard the pairing and inflate the interval.
"""

import random

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

SCORE_LABELS = [0, 1, 2]
DEFAULT_SEED = 12345
DEFAULT_BOOTSTRAP = 4000


def confusion_matrix_3x3(human, auto):
    """counts[(human_score, auto_score)] -> n."""
    counts = {(h, a): 0 for h in SCORE_LABELS for a in SCORE_LABELS}
    for h, a in zip(human, auto):
        counts[(h, a)] += 1
    return counts


def agreement_rates(human, auto):
    """Exact / off-by-one / off-by-two shares, plus directional bias."""
    n = len(human)
    if not n:
        return {}
    diffs = [a - h for h, a in zip(human, auto)]
    return {
        "n": n,
        "exact": sum(1 for d in diffs if d == 0) / n,
        "off_by_one": sum(1 for d in diffs if abs(d) == 1) / n,
        "off_by_two": sum(1 for d in diffs if abs(d) == 2) / n,
        "auto_too_low": sum(1 for d in diffs if d < 0),
        "auto_too_high": sum(1 for d in diffs if d > 0),
    }


def kappa(human, auto, weights=None):
    """Cohen's kappa. weights='linear' for the ordinal-aware variant."""
    if len(set(human)) < 2 and len(set(auto)) < 2:
        return 0.0
    return float(cohen_kappa_score(human, auto, labels=SCORE_LABELS, weights=weights))


def spearman(scores, human):
    if len(set(scores)) < 2 or len(set(human)) < 2:
        return 0.0
    rho = spearmanr(scores, human).correlation
    return 0.0 if rho != rho else float(rho)  # NaN guard


def paired_bootstrap_spearman_delta(human, metric_a, metric_b,
                                    n_boot=DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED):
    """Bootstrap CI for rho(metric_a, human) - rho(metric_b, human).

    Positive interval excluding 0 => metric_a tracks human judgment better.
    An interval containing 0 means the two metrics are not separable at this
    sample size, which is a legitimate result and must not be tuned away.
    """
    n = len(human)
    rng = random.Random(seed)
    observed = spearman(metric_a, human) - spearman(metric_b, human)

    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]  # one resample, both metrics
        h = [human[i] for i in idx]
        if len(set(h)) < 2:
            continue
        a = spearman([metric_a[i] for i in idx], h)
        b = spearman([metric_b[i] for i in idx], h)
        deltas.append(a - b)

    if len(deltas) < 50:
        return {"delta": observed, "lo": float("nan"), "hi": float("nan"), "n_boot": len(deltas)}
    deltas.sort()
    return {
        "delta": observed,
        "lo": deltas[int(0.025 * len(deltas))],
        "hi": deltas[int(0.975 * len(deltas))],
        "n_boot": len(deltas),
    }


def verdict(ci):
    """Turn a paired-bootstrap interval into a decision."""
    if ci["lo"] != ci["lo"]:  # NaN
        return "undetermined"
    if ci["lo"] > 0:
        return "a_better"
    if ci["hi"] < 0:
        return "b_better"
    return "tie"


def _score(sim, hi, lo):
    return 2 if sim > hi else (1 if sim > lo else 0)


def oracle_threshold_ceiling(sims, human, grid=None):
    """Best achievable 3-class accuracy from similarity alone.

    Thresholds are fitted directly on the labels being scored, so this is an
    optimistic upper bound - it exists to show that threshold tuning cannot
    close the gap, not to propose thresholds.
    """
    grid = grid or [i / 100 for i in range(-20, 101, 5)]
    best = (0.0, None, None)
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            acc = sum(1 for s, h in zip(sims, human) if h == _score(s, hi, lo)) / len(human)
            if acc > best[0]:
                best = (acc, hi, lo)
    majority = max(sum(1 for h in human if h == c) for c in SCORE_LABELS) / len(human)
    return {"oracle_acc": best[0], "hi": best[1], "lo": best[2], "majority_acc": majority}


def cv_threshold_tuning(sims, human, k=5, seed=0, grid=None):
    """Honest held-out estimate of what threshold tuning actually buys."""
    grid = grid or [i / 100 for i in range(5, 96, 5)]
    idx = list(range(len(human)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]

    predicted, actual = [], []
    for i in range(k):
        train = [j for f_i, fold in enumerate(folds) if f_i != i for j in fold]
        best = (-2.0, None, None)
        for lo in grid:
            for hi in grid:
                if hi <= lo:
                    continue
                scored = [_score(sims[j], hi, lo) for j in train]
                kap = kappa([human[j] for j in train], scored, weights="linear")
                if kap > best[0]:
                    best = (kap, hi, lo)
        _, hi, lo = best
        for j in folds[i]:
            predicted.append(_score(sims[j], hi, lo))
            actual.append(human[j])

    return {
        "acc": sum(1 for p, a in zip(predicted, actual) if p == a) / len(actual),
        "kappa_linear": kappa(actual, predicted, weights="linear"),
    }


def summarize_granularity(human, auto, sims, keystrokes):
    """Every headline statistic for one granularity."""
    rates = agreement_rates(human, auto)
    return {
        **rates,
        "kappa": kappa(human, auto),
        "kappa_linear": kappa(human, auto, weights="linear"),
        "rho_embed": spearman(sims, human),
        "rho_keystrokes": spearman(keystrokes, human),
        "paired_ci": paired_bootstrap_spearman_delta(human, keystrokes, sims),
    }


if __name__ == "__main__":
    # Perfect agreement and a known-bad scorer, as a sanity floor.
    human = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    print("kappa(self) =", kappa(human, human))
    print("kappa(inverted) =", round(kappa(human, [2, 1, 0] * 3), 3))
    print("rates(self) =", agreement_rates(human, human))
    ci = paired_bootstrap_spearman_delta(human, human, [2, 1, 0] * 3, n_boot=500, seed=1)
    print("paired CI (good vs inverted) =",
          f"delta={ci['delta']:+.3f} [{ci['lo']:+.3f},{ci['hi']:+.3f}] -> {verdict(ci)}")
