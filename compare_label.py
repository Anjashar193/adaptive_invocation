"""Compare hand labels against the automatic scorer.

Thin CLI over src/label_join.py (recovering the labelled rows) and
src/agreement.py (the statistics). Previously this script read columns that
hand_labels.csv no longer has, so it crashed with KeyError: 'usefulness_auto'.
"""

from src.agreement import (SCORE_LABELS, agreement_rates, confusion_matrix_3x3,
                           kappa, summarize_granularity, verdict)
from src.label_join import load_joined
from src.metrics import keystrokes_saved

GRANULARITIES = ["next_word", "partial_word", "phrase", "sentence"]

VERDICT_TEXT = {
    "a_better": "keystrokes track humans better",
    "b_better": "embeddings track humans better",
    "tie": "not separable at this n",
    "undetermined": "insufficient data",
}


def _columns(rows):
    human = [int(float(j["label"]["manual_score"])) for j in rows]
    auto = [int(float(j["label"]["auto_score"])) for j in rows]
    sims = [j["sim"] for j in rows]
    keys = [keystrokes_saved(j["result"]["prediction"], j["result"]["reference"])
            for j in rows]
    return human, auto, sims, keys


def main():
    joined, unmatched = load_joined()
    total = len(joined) + len(unmatched)
    print(f"Joined {len(joined)}/{total} labels to their result rows.")
    if unmatched:
        print(f"  unmatched ids: {sorted(r['id'] for r in unmatched)}")
    ambiguous = sum(1 for j in joined if j["match_count"] > 1)
    print(f"  {ambiguous} keys match several result rows (all with equal usefulness)\n")

    print("Agreement with the automatic scorer")
    print(f"{'granularity':<14}{'n':>5}{'exact':>9}{'kappa':>9}{'k-linear':>10}"
          f"{'rho_emb':>9}{'rho_ks':>9}")
    print("-" * 65)
    for granularity in GRANULARITIES:
        rows = [j for j in joined if j["label"]["granularity"] == granularity]
        if not rows:
            continue
        human, auto, sims, keys = _columns(rows)
        stats = summarize_granularity(human, auto, sims, keys)
        print(f"{granularity:<14}{stats['n']:>5}{stats['exact'] * 100:>8.1f}%"
              f"{stats['kappa']:>+9.3f}{stats['kappa_linear']:>+10.3f}"
              f"{stats['rho_embed']:>+9.3f}{stats['rho_keystrokes']:>+9.3f}")

    human, auto, _, _ = _columns(joined)
    overall = agreement_rates(human, auto)
    print(f"{'OVERALL':<14}{overall['n']:>5}{overall['exact'] * 100:>8.1f}%"
          f"{kappa(human, auto):>+9.3f}{kappa(human, auto, weights='linear'):>+10.3f}")

    print(f"\nDisagreement shape: {overall['off_by_one'] * 100:.1f}% off by one, "
          f"{overall['off_by_two'] * 100:.1f}% off by two.")
    print(f"Direction: auto scored too LOW {overall['auto_too_low']} times, "
          f"too HIGH {overall['auto_too_high']} times.")
    print("Linear-weighted kappa is the headline number - most disagreement is")
    print("one point apart, which unweighted kappa penalises as harshly as a")
    print("complete reversal.")

    print("\nWhich metric tracks human judgment better?")
    print("(paired bootstrap on the Spearman difference, keystrokes - embeddings)")
    print(f"{'granularity':<14}{'delta':>9}{'95% CI':>22}   verdict")
    print("-" * 68)
    for granularity in GRANULARITIES:
        rows = [j for j in joined if j["label"]["granularity"] == granularity]
        if not rows:
            continue
        human, auto, sims, keys = _columns(rows)
        stats = summarize_granularity(human, auto, sims, keys)
        ci = stats["paired_ci"]
        print(f"{granularity:<14}{ci['delta']:>+9.3f}   [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
              f"   {VERDICT_TEXT[verdict(ci)]}")

    print("\nConfusion matrix (rows = human, cols = automatic):")
    human, auto, _, _ = _columns(joined)
    counts = confusion_matrix_3x3(human, auto)
    print(f"{'':>12}" + "".join(f"{'auto=' + str(a):>10}" for a in SCORE_LABELS))
    for h in SCORE_LABELS:
        print(f"human={h:<6}" + "".join(f"{counts[(h, a)]:>10}" for a in SCORE_LABELS))


if __name__ == "__main__":
    main()
