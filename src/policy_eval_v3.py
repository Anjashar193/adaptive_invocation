"""Phase 2a: compare invocation policies on the Stage A6 results (phase1_results_v3.csv).

Same simulation and staleness rule as policy_eval.py, with these changes:
  * arrival time uses latency_top1_ms (time to the FIRST suggestion, what a user waits
    for); `top5` as 2nd argument uses the 5-candidate time instead, for comparison
  * more pause thresholds, plus combined rules "boundary + mid-word T":
    fire right after every space, and inside a word only after a pause longer than T
  * uncertainty: 95 % interval over MESSAGES (bootstrap, 1000 resamples), since the
    5 cut points of a message are not independent
  * a "word-level mix" table that pools partial_word (mid-word position) and next_word
    (word-start position) with the weights of real typing: a message has about
    <letters per word> mid-word keystrokes for every space
  * a PASS column that applies the selection rule below (thresholds are constants)

Selection rule (proposed, not yet confirmed by the supervisor): a policy PASSES if
  calls per 100 keystrokes <= MAX_CALLS_PCT of always-invoke's,
  missed <= MAX_MISSED_PCT, and value per call >= MIN_VPC_RATIO x always-invoke's.

Run from the repo root:
    uv run python -m src.policy_eval_v3 [aalto|0|1|2] [top5]
Writes results/policy_eval_v3_summary.csv
"""
import os
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

import json

RESULTS = os.environ.get("RESULTS", "phase1_results_v3.csv")
CHOICE = "metric_choice.json"
PACE = "amazonqac_pace_quantiles.csv"
SUMMARY = "results/policy_eval_v3_summary.csv"
SEED = 42
REPEATS = 10
N_BOOT = 1000
GRANS = ["partial_word", "next_word", "phrase", "sentence"]
GOOD_SIM = 0.4                       # provisional: a "good" similarity-scored suggestion
README_METRIC = {"next_word": "keystrokes_saved", "partial_word": "keystrokes_saved",
                 "phrase": "keystrokes_saved", "sentence": "semantic_similarity"}
MAX_CALLS_PCT, MAX_MISSED_PCT, MIN_VPC_RATIO = 50.0, 25.0, 0.95
MIX = "word-level mix"
N_STAT = 7                           # rows, calls, value, rate, good, missed, base value


def const(x):
    return lambda g, sp: np.full(len(g), x)


def build_policies():
    p = {
        "always": (const(0.0), lambda g, sp, n: np.ones(len(g), dtype=bool)),
        "every 3 keys": (const(0.0), lambda g, sp, n: n % 3 == 0),
        "every 5 keys": (const(0.0), lambda g, sp, n: n % 5 == 0),
    }
    for t in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2):
        p[f"pause {t}s"] = (const(t), lambda g, sp, n, t=t: g > t)
    p["word boundary"] = (const(0.0), lambda g, sp, n: sp)
    for t in (0.4, 0.6, 1.0):
        p[f"boundary + mid-word {t}s"] = (
            lambda g, sp, t=t: np.where(sp, 0.0, t),
            lambda g, sp, n, t=t: sp | (g > t))
    return p


def chosen_metric(choice, g):
    entry = choice.get(g, {})
    names = {"keystrokes": "keystrokes_saved", "embeddings": "semantic_similarity"}
    if isinstance(entry, dict) and entry.get("metric") in names:
        return names[entry["metric"]]
    print(f"  ! could not read the metric for {g}; using the README table value. Entry was: {entry}")
    return README_METRIC[g]


def metrics_from(tot, lat):
    """tot: (n_policy, N_STAT) sums -> dict of arrays per policy."""
    rows, calls, value, rate, good, missed, base = (tot[:, i] for i in range(N_STAT))
    with np.errstate(divide="ignore", invalid="ignore"):
        calls100 = 100 * rate / rows
        return dict(calls100=calls100, modelsec=calls100 * lat,
                    vpc=np.where(calls > 0, value / calls, np.nan),
                    kept=100 * value / base, missed=100 * missed / good)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "aalto"
    use_top5 = "top5" in sys.argv[2:]
    use_aalto = arg == "aalto"
    shape = load_gap_shape() if use_aalto else None
    sigma, bmult = SCENARIOS[0 if use_aalto else int(arg)]
    rng = np.random.default_rng(SEED)
    levels, values = load_pace_quantiles(PACE)
    with open(CHOICE) as f:
        choice = json.load(f)
    policies = build_policies()
    names = list(policies)
    pidx = {n: i for i, n in enumerate(names)}

    df = pd.read_csv(RESULTS)
    df = df[df["system"] == "qwen"].copy()
    lat_col = "latency_ms" if use_top5 else "latency_top1_ms"
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df["semantic_sim"] = pd.to_numeric(df.get("semantic_sim"), errors="coerce")
    msg_ids = sorted(df["message_id"].unique())
    midx = {m: i for i, m in enumerate(msg_ids)}
    nm = len(msg_ids)

    # letters per word -> weight of a mid-word example relative to a word-start example
    ratio = float((df["prefix"].str.len() / df["prefix_words"].clip(lower=1)).mean()) - 1.0
    weights = {"partial_word": ratio, "next_word": 1.0}

    print("Typing model: Aalto gap shape (real) + AmazonQAC pace (real)" if use_aalto else
          f"Typing model: modelled style {arg} (sigma={sigma}, boundary x{bmult}) - robustness check")
    print(f"Source: {RESULTS}, {nm} messages, {len(df)} Qwen examples; latency column: {lat_col}")
    print(f"Seed {SEED}, {REPEATS} simulated typists per example; 95 % intervals over messages ({N_BOOT} resamples)")
    print(f"Pass rule: calls <= {MAX_CALLS_PCT:.0f} % of always, missed <= {MAX_MISSED_PCT:.0f} %, "
          f"value/call >= {MIN_VPC_RATIO} x always")
    print(f"Word-level mix weights: mid-word example x {ratio:.2f}, word-start example x 1")

    groups = GRANS + [MIX]
    A = {g: np.zeros((nm, len(names), N_STAT)) for g in groups}
    lat_sum = {g: 0.0 for g in groups}
    lat_w = {g: 0.0 for g in groups}
    lat_top5 = {g: [] for g in GRANS}
    used = {g: 0 for g in GRANS}
    ks_ok = {}

    for g in GRANS:
        sub = df[df["granularity"] == g]
        metric = chosen_metric(choice, g)
        use_ks = metric == "keystrokes_saved"
        ks_ok[g] = metric
        for row in sub.itertuples():
            prefix = str(row.prefix) if pd.notna(row.prefix) else ""
            ref = str(row.reference) if pd.notna(row.reference) else ""
            pred = str(row.prediction) if pd.notna(row.prediction) else ""
            lat = getattr(row, lat_col)
            if not prefix or not ref or pd.isna(lat):
                continue
            sim = row.semantic_sim
            if not use_ks and pd.isna(sim):
                continue
            L = lat / 1000
            ks0 = keystrokes_saved(pred, ref)
            mi = midx[row.message_id]
            used[g] += 1
            lat_sum[g] += L
            lat_w[g] += 1
            w = weights.get(g)
            if w is not None:
                lat_sum[MIX] += L * w
                lat_w[MIX] += w
            if g == "partial_word":
                full, dec = prefix + ref, len(prefix) - 1
            else:
                full, dec = prefix + " " + ref, len(prefix)
            is_space = np.array([c == " " for c in full[:-1]])

            for _ in range(REPEATS):
                pace = sample_pace(rng, levels, values)
                tl = (simulate_typing_empirical(full, pace, shape, rng) if use_aalto
                      else simulate_typing(full, pace, sigma, bmult, rng))
                times = np.array([t for t, _ in tl])
                gaps = np.diff(times)
                n_typed = np.arange(1, len(gaps) + 1)
                outs = []
                for name in names:
                    wait_fn, rule = policies[name]
                    fires = rule(gaps, is_space, n_typed)
                    invoked = bool(fires[dec])
                    rate = float(fires[: dec + 1].mean())
                    value = 0.0
                    if invoked:
                        arrival = times[dec] + wait_fn(gaps, is_space)[dec] + L
                        k = int(np.searchsorted(times[dec + 1:], arrival, side="right"))
                        if ks0 >= k:                            # staleness rule
                            value = float(ks0 - k) if use_ks else float(sim)
                    outs.append((invoked, rate, value))
                base_v = outs[0][2]
                good = base_v > 0 if use_ks else base_v >= GOOD_SIM
                targets = [(g, 1.0)] + ([(MIX, w)] if w is not None else [])
                for grp, wt in targets:
                    for j, (invoked, rate, value) in enumerate(outs):
                        a = A[grp][mi, j]
                        a[0] += wt
                        a[1] += wt * invoked
                        a[2] += wt * value
                        a[3] += wt * rate
                        a[4] += wt * good
                        a[5] += wt * (good and value <= 0)
                        a[6] += wt * base_v
        lat_top5[g] = sub["latency_ms"].mean() / 1000 if "latency_ms" in sub else float("nan")

    summary = []
    boot_rng = np.random.default_rng(SEED + 1)
    for grp in groups:
        lat = lat_sum[grp] / max(lat_w[grp], 1e-9)
        m = metrics_from(A[grp].sum(0), lat)
        bm = []
        for _ in range(N_BOOT):
            idx = boot_rng.integers(0, nm, nm)
            bm.append(metrics_from(A[grp][idx].sum(0), lat)["missed"])
        lo, hi = np.nanpercentile(np.array(bm), [2.5, 97.5], axis=0)
        if grp == MIX:
            head = f"{MIX} (partial_word + next_word, keystrokes saved)"
        else:
            head = (f"{grp}: metric = {ks_ok[grp]}; {used[grp]} examples; mean latency used "
                    f"{lat:.2f}s (5-candidate time {lat_top5[grp]:.2f}s)")
        print(f"\n== {head} ==")
        print(f"{'policy':<26}{'calls/100 keys':>15}{'model-s/100 keys':>18}{'value/call':>12}"
              f"{'value kept %':>14}{'missed %':>10}{'  95% interval':>16}  pass")
        vpc0 = m["vpc"][0]
        for j, name in enumerate(names):
            ok = (j != 0 and m["calls100"][j] <= MAX_CALLS_PCT and m["missed"][j] <= MAX_MISSED_PCT
                  and not np.isnan(m["vpc"][j]) and m["vpc"][j] >= MIN_VPC_RATIO * vpc0)
            print(f"{name:<26}{m['calls100'][j]:>15.1f}{m['modelsec'][j]:>18.1f}{m['vpc'][j]:>12.2f}"
                  f"{m['kept'][j]:>14.1f}{m['missed'][j]:>10.1f}{f'[{lo[j]:.0f}-{hi[j]:.0f}]':>16}  {'PASS' if ok else ''}")
            summary.append(dict(group=grp, policy=name, calls100=m["calls100"][j], modelsec100=m["modelsec"][j],
                                value_per_call=m["vpc"][j], value_kept=m["kept"][j], missed=m["missed"][j],
                                missed_lo=lo[j], missed_hi=hi[j], passes=bool(ok)))
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    print(f"\nSaved {SUMMARY}")


if __name__ == "__main__":
    main()