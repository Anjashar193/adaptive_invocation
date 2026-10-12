"""Phase 2a pilot: compare invocation policies on the frozen Phase 1 predictions.

For every Qwen example (a cut point in an OASST2 message) we simulate the user
typing, let each policy decide whether to call the model at that moment, and
score the suggestion WHEN IT ARRIVES (staleness rule, enforced in code):

  * while the model runs, the user keeps typing the true text (k characters)
  * the suggestion is still valid only if its first k characters match what was
    typed, i.e. keystrokes_saved(prediction, reference) >= k; otherwise it is worth 0
  * value on arrival: keystroke granularities -> characters still saved (ks - k)
                      similarity granularity (per metric_choice.json) -> similarity

The four numbers (Step 10), per granularity and policy:
  cost             calls per 100 keystrokes AND model-seconds per 100 keystrokes
  quality          value delivered per call (units depend on the metric)
  invocation rate  = calls per 100 keystrokes
  missed           share of the suggestions that always-invoke delivered which this
                   policy did not deliver (never called, or arrived stale)

Typing model: pace per query from AmazonQAC (real); gaps inside a query either
modelled (styles 0/1/2, robustness checks) or the real Aalto gap shape ("aalto").

Run from the repo root:   uv run python -m src.policy_eval [0|1|2|aalto]
  0 = steady typist, 1 = medium (default), 2 = bursty typist, aalto = real gap shape
"""
import json
import sys

import numpy as np
import pandas as pd

try:
    from src.metrics import keystrokes_saved
    from src.typing_sim import SCENARIOS, load_pace_quantiles, sample_pace, simulate_typing
    from src.gap_shape import load_gap_shape, simulate_typing_empirical
except ImportError:  # run as a plain script from inside src/
    from metrics import keystrokes_saved
    from typing_sim import SCENARIOS, load_pace_quantiles, sample_pace, simulate_typing
    from gap_shape import load_gap_shape, simulate_typing_empirical

RESULTS = "phase1_results_v2.csv"      # frozen Phase 1 results - read only
CHOICE = "metric_choice.json"
PACE = "amazonqac_pace_quantiles.csv"
SEED = 42
REPEATS = 10                           # simulated typists per example
GRANS = ["partial_word", "next_word", "phrase", "sentence"]
GOOD_SIM = 0.4                         # provisional: what counts as a "good" similarity-scored suggestion
README_METRIC = {"next_word": "keystrokes_saved", "partial_word": "keystrokes_saved",
                 "phrase": "keystrokes_saved", "sentence": "semantic_similarity"}
EXPECTED_KS_RATE = {"next_word": 25.9, "partial_word": 9.9, "phrase": 7.1, "sentence": 3.9}  # from the metric plan

# name: (seconds to wait after the keystroke before calling, rule over all keystrokes)
# rule(gap_after_key, key_is_space, number_of_keys_typed_so_far) -> bool array
POLICIES = {
    "always":        (0.0, lambda g, sp, n: np.ones(len(g), dtype=bool)),
    "every 3 keys":  (0.0, lambda g, sp, n: n % 3 == 0),
    "every 5 keys":  (0.0, lambda g, sp, n: n % 5 == 0),
    "pause 0.3s":    (0.3, lambda g, sp, n: g > 0.3),
    "pause 0.6s":    (0.6, lambda g, sp, n: g > 0.6),
    "pause 1.0s":    (1.0, lambda g, sp, n: g > 1.0),
    "word boundary": (0.0, lambda g, sp, n: sp),
}


def chosen_metric(choice, g):
    entry = choice.get(g, {})
    names = {"keystrokes": "keystrokes_saved", "embeddings": "semantic_similarity"}
    if isinstance(entry, dict) and entry.get("metric") in names:
        return names[entry["metric"]]
    print(f"  ! could not read the metric for {g} from {CHOICE}; using the README table value. Entry was: {entry}")
    return README_METRIC[g]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    use_aalto = arg == "aalto"
    shape = load_gap_shape() if use_aalto else None
    style = 0 if use_aalto else int(arg)
    sigma, bmult = SCENARIOS[style]
    rng = np.random.default_rng(SEED)
    levels, values = load_pace_quantiles(PACE)
    with open(CHOICE) as f:
        choice = json.load(f)

    df = pd.read_csv(RESULTS)
    df = df[df["system"] == "qwen"].copy()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "semantic_sim" in df:
        df["semantic_sim"] = pd.to_numeric(df["semantic_sim"], errors="coerce")

    print("Typing style: Aalto gap shape (real) with AmazonQAC pace (real)" if use_aalto else
          f"Typing style {style}: sigma={sigma}, boundary x{bmult}  (modelled gaps - robustness check)")
    print(f"Source: {RESULTS} (Qwen rows), seed {SEED}, {REPEATS} simulated typists per example")

    for g in GRANS:
        sub = df[df["granularity"] == g]
        metric = chosen_metric(choice, g)
        use_ks = metric == "keystrokes_saved"
        lat_mean = sub["latency_ms"].mean() / 1000
        acc = {p: dict(rows=0, calls=0, value=0.0, rate=0.0, good=0, missed=0) for p in POLICIES}
        base_total, ks_sum, len_sum, n_ex = 0.0, 0, 0, 0

        for row in sub.itertuples():
            prefix = str(row.prefix) if pd.notna(row.prefix) else ""
            ref = str(row.reference) if pd.notna(row.reference) else ""
            pred = str(row.prediction) if pd.notna(row.prediction) else ""
            if not prefix or not ref or pd.isna(row.latency_ms):
                continue
            sim = getattr(row, "semantic_sim", float("nan"))
            if not use_ks and pd.isna(sim):
                continue
            L = row.latency_ms / 1000
            ks0 = keystrokes_saved(pred, ref)
            ks_sum += ks0
            len_sum += len(ref)
            n_ex += 1

            # the moment the policy must decide: after the last typed char (mid-word) or after the space
            if g == "partial_word":
                full, dec = prefix + ref, len(prefix) - 1
            else:
                full, dec = prefix + " " + ref, len(prefix)

            for _ in range(REPEATS):
                pace = sample_pace(rng, levels, values)
                if use_aalto:
                    tl = simulate_typing_empirical(full, pace, shape, rng)
                else:
                    tl = simulate_typing(full, pace, sigma, bmult, rng)
                times = np.array([t for t, _ in tl])
                gaps = np.diff(times)
                is_space = np.array([c == " " for c in full[:-1]])
                n_typed = np.arange(1, len(gaps) + 1)

                outcome = {}
                for name, (wait, rule) in POLICIES.items():
                    fires = rule(gaps, is_space, n_typed)
                    invoked = bool(fires[dec])
                    rate = float(fires[: dec + 1].mean())
                    value = 0.0
                    if invoked:
                        arrival = times[dec] + wait + L
                        k = int(np.searchsorted(times[dec + 1:], arrival, side="right"))
                        if ks0 >= k:                       # staleness rule
                            value = float(ks0 - k) if use_ks else float(sim)
                    outcome[name] = (invoked, rate, value)

                base_v = outcome["always"][2]
                good = base_v > 0 if use_ks else base_v >= GOOD_SIM
                base_total += base_v
                for name, (invoked, rate, value) in outcome.items():
                    a = acc[name]
                    a["rows"] += 1
                    a["calls"] += int(invoked)
                    a["value"] += value
                    a["rate"] += rate
                    if good:
                        a["good"] += 1
                        a["missed"] += int(value <= 0)

        unit = "chars still saved on arrival" if use_ks else "similarity if still valid"
        print(f"\n== {g}: metric = {metric} ({unit}); {n_ex} examples ==")
        print(f"   check: keystroke rate with zero delay = {100 * ks_sum / max(len_sum, 1):.1f}%"
              f"  (metric plan: {EXPECTED_KS_RATE[g]}%)   mean latency {lat_mean:.2f}s")
        print(f"{'policy':<15}{'calls/100 keys':>15}{'model-s/100 keys':>18}{'value/call':>12}"
              f"{'value kept %':>14}{'missed %':>10}")
        for name, a in acc.items():
            rate100 = 100 * a["rate"] / max(a["rows"], 1)
            vpc = a["value"] / a["calls"] if a["calls"] else float("nan")
            kept = 100 * a["value"] / base_total if base_total else float("nan")
            missed = 100 * a["missed"] / a["good"] if a["good"] else float("nan")
            print(f"{name:<15}{rate100:>15.1f}{rate100 * lat_mean:>18.1f}{vpc:>12.2f}{kept:>14.1f}{missed:>10.1f}")


if __name__ == "__main__":
    main()
