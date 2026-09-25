"""Post-labeling power re-check (PHASE2_METRIC_PLAN.md Part 3.3, item #7):
"partial_word >= 80%; if not, extend that granularity before deciding."

Method, matching the plan's own Part 2.4 methodology: bootstrap-resample the
ACTUAL final labeled data (with replacement, same size as the real sample)
many times, and check what fraction of resamples produce the SAME verdict
as the real, full-data decision. This is the achieved power at the sample
size we actually ended up with - not the pre-labeling projection, which was
a simulation guess before any real data existed.
"""

import csv
import random

try:
    from agreement import paired_bootstrap_spearman_delta, verdict
    from metrics import keystrokes_saved, semantic_similarity, trim_prediction
except ImportError:
    from src.agreement import paired_bootstrap_spearman_delta, verdict
    from src.metrics import keystrokes_saved, semantic_similarity, trim_prediction

GRANULARITIES = ["next_word", "partial_word", "phrase", "sentence"]
N_TRIALS = 60  # matches the plan's own convention (Part 2.4: "60 trials per size")
TARGET_POWER = 0.80


def load_matches_data(labels_csv, granularity):
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["axis"] == "matches" and r["granularity"] == granularity]

    cache = {}
    human, keystrokes, sims = [], [], []
    for r in rows:
        pred, ref = r["prediction"], r["reference"]
        human.append(int(r["score"]))
        keystrokes.append(keystrokes_saved(pred, ref))
        key = (granularity, pred, ref)
        if key not in cache:
            trimmed = trim_prediction(pred, ref, granularity)
            cache[key] = semantic_similarity(trimmed, ref)
        sims.append(cache[key])
    return human, keystrokes, sims


def power_recheck(labels_csv="control_labels.csv", n_trials=N_TRIALS, seed=0, trial_n_boot=300):
    print(f"Achieved power at final sample size ({n_trials} bootstrap trials per granularity)\n")
    print(f"{'granularity':<14}{'n':<6}{'real verdict':<14}{'power':<8}{'status'}")
    print("-" * 60)

    results = {}
    for granularity in GRANULARITIES:
        human, keystrokes, sims = load_matches_data(labels_csv, granularity)
        n = len(human)
        if n < 10:
            print(f"{granularity:<14}{n:<6}(too few labels to check)")
            continue

        # the REAL, reported decision uses full precision (default n_boot=4000)
        real_ci = paired_bootstrap_spearman_delta(human, keystrokes, sims)
        real_verdict = verdict(real_ci)

        rng = random.Random(seed)
        matches = 0
        for _ in range(n_trials):
            idx = [rng.randrange(n) for _ in range(n)]
            h = [human[i] for i in idx]
            k = [keystrokes[i] for i in idx]
            s = [sims[i] for i in idx]
            if len(set(h)) < 2:
                continue
            # trials use a smaller n_boot - this is a power SIMULATION, not
            # the reported statistic itself, so reduced precision here is standard
            trial_ci = paired_bootstrap_spearman_delta(h, k, s, n_boot=trial_n_boot, seed=seed + 1)
            trial_verdict = verdict(trial_ci)
            if trial_verdict == real_verdict:
                matches += 1

        power = matches / n_trials
        status = "OK" if power >= TARGET_POWER else f"BELOW {TARGET_POWER:.0%} TARGET"
        results[granularity] = {"n": n, "verdict": real_verdict, "power": power}
        print(f"{granularity:<14}{n:<6}{real_verdict:<14}{power:<8.1%}{status}")

    return results


if __name__ == "__main__":
    power_recheck()
