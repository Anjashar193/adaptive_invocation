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
EXPECTED_KS_RATE = {"next_word": 25.9, "partial_word": 9.9, "phrase": 7.1, "sentence": 3.9}

PAUSE_GRID = [0.2, 0.4, 0.6, 0.8, 1.2]          # requested grid, replaces {0.3,0.6,1.0}


def make_policies():
    pols = {
        "always":        (0.0, lambda g, sp, n, ks: np.ones(len(g), dtype=bool)),
        "every 3 keys":  (0.0, lambda g, sp, n, ks: n % 3 == 0),
        "every 5 keys":  (0.0, lambda g, sp, n, ks: n % 5 == 0),
        "word boundary": (0.0, lambda g, sp, n, ks: sp),
    }
    for t in PAUSE_GRID:
        pols[f"pause {t}s"] = (t, (lambda t_: (lambda g, sp, n, ks: g > t_))(t))
    # NEW: fire mid-word when the user has paused AND is inside a word
    # (>=2 keys since the last space). Gap threshold matches the middle of the grid.
    pols["mid-word 0.6s"] = (0.6, lambda g, sp, n, ks: (g > 0.6) & (ks >= 2))
    return pols


def chosen_metric(choice, g):
    entry = choice.get(g, {})
    names = {"keystrokes": "keystrokes_saved", "embeddings": "semantic_similarity"}
    if isinstance(entry, dict) and entry.get("metric") in names:
        return names[entry["metric"]]
    print(f"  ! could not read the metric for {g} from {CHOICE}; using the README table value.")
    return README_METRIC[g]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "aalto"
    use_aalto = arg == "aalto"
    shape = load_gap_shape() if use_aalto else None
    style = 0 if use_aalto else int(arg)
    sigma, bmult = SCENARIOS[style]
    POLICIES = make_policies()
    rng = np.random.default_rng(SEED)
    levels, values = load_pace_quantiles(PACE)
    with open(CHOICE) as f:
        choice = json.load(f)

    df = pd.read_csv(RESULTS)
    df = df[df["system"] == "qwen"].copy()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "semantic_sim" in df:
        df["semantic_sim"] = pd.to_numeric(df["semantic_sim"], errors="coerce")

    print("Typing model: Aalto gap shape (real) x AmazonQAC pace (real)" if use_aalto else
          f"Typing model: modelled style {style} (sigma={sigma}, boundary x{bmult}) x AmazonQAC pace")
    print(f"Source: {RESULTS} (Qwen rows), seed {SEED}, {REPEATS} simulated typists per example")
    print(f"Pause grid: {PAUSE_GRID}s (replaces the old fixed 0.3/0.6/1.0s)")
    print(LIMITATIONS)

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

            if g == "partial_word":
                full, dec = prefix + ref, len(prefix) - 1
            else:
                full, dec = prefix + " ", len(prefix)

            for _ in range(REPEATS):
                pace = sample_pace(rng, levels, values)
                tl = (simulate_typing_empirical(full, pace, shape, rng) if use_aalto
                      else simulate_typing(full, pace, sigma, bmult, rng))
                times = np.array([t for t, _ in tl])
                gaps = np.diff(times)
                is_space = np.array([c == " " for c in full[:-1]])
                n_typed = np.arange(1, len(gaps) + 1)
                # keys typed since the most recent space, at each gap position
                # (kss[i] = length of the current word-run after keystroke i+1)
                kss = np.empty(len(is_space), dtype=int)
                run = 0
                for i, sp in enumerate(is_space):
                    run = 0 if sp else run + 1
                    kss[i] = run

                outcome = {}
                for name, (wait, rule) in POLICIES.items():
                    fires = rule(gaps, is_space, n_typed, kss)
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
        print(f"{'policy':<16}{'calls/100 dp':>13}{'calls/100 keys':>15}{'model-s/100 keys':>17}"
              f"{'value/call':>11}{'value kept %':>13}{'missed %':>10}")
        for name, a in acc.items():
            cr100 = 100 * a["calls"] / max(a["rows"], 1)          # per decision point
            kr100 = 100 * a["rate"] / max(a["rows"], 1)           # per keystroke fired-at
            vpc = a["value"] / a["calls"] if a["calls"] else float("nan")
            kept = 100 * a["value"] / base_total if base_total else float("nan")
            missed = 100 * a["missed"] / a["good"] if a["good"] else float("nan")
            print(f"{name:<16}{cr100:>13.1f}{kr100:>15.1f}{kr100 * lat_mean:>17.1f}"
                  f"{vpc:>11.2f}{kept:>13.1f}{missed:>10.1f}")


if __name__ == "__main__":
    main()