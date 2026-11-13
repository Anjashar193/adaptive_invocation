"""Evaluate the FROZEN policy (thresholds in src/policy.py) on one dataset. Nothing is tuned here.

Used for the second-dataset check (Stage A8): run it on OASST2 and on AESLC and compare.

    RESULTS=phase1_results_v3.csv    uv run python -m src.eval_frozen    # OASST2 (240 messages)
    RESULTS=phase1_results_aeslc.csv uv run python -m src.eval_frozen    # AESLC (120 messages)

It uses the same simulated typists, staleness rule and four numbers as src/learned_policy.py
(same seeds), but all messages: there is no dev/test split, because no choice is made.
For each granularity the frozen rule decides at the cut point of every example:
    sentence       call if words typed >= SENTENCE_MIN_WORDS
    partial_word   call if the word has >= MIN_LETTERS letters
    word-level mix call at a word start, or mid-word once the word has >= MIN_LETTERS letters
Reported: calls (% of decision points), missed (% of the suggestions always-invoke delivered),
random skipping at the same calls (missed = 100 - calls), value per call, and the share of
bootstrap resamples (over messages) in which the policy meets the proposed selection rule
(calls <= 50 %, missed <= 25 %, value per call >= 0.95 x always-invoke).
next_word and phrase are not listed: the frozen policy calls next_word at every word start.
Writes results/frozen_policy_<dataset>.csv
"""
import json
import os

import numpy as np
import pandas as pd

try:
    from src import learned_policy as lp
    from src import policy as pol
except ImportError:  # run as a plain script from inside src/
    import learned_policy as lp
    import policy as pol

GROUPS = ["sentence", lp.MIX, "partial_word"]


def frozen_fire(S, idx, grp):
    X = S["feat"][idx]

    def col(name):
        return X[:, lp.ALL_FEATURES.index(name)]
    kss, at_space, n_words = col("kss"), col("at_space") > 0.5, col("n_words")
    if grp == "sentence":
        return n_words >= pol.SENTENCE_MIN_WORDS
    if grp == "partial_word":
        return kss >= pol.MIN_LETTERS
    return at_space | (kss >= pol.MIN_LETTERS)          # word-level mix


def numbers(a, alw):
    """a, alw: (n_messages, N_STAT) sums -> calls %, missed %, value per call, always value per call."""
    s, t = a.sum(0), alw.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (100 * s[1] / s[0], 100 * s[3] / s[2],
                s[4] / s[1] if s[1] > 0 else np.nan, t[4] / t[1] if t[1] > 0 else np.nan)


def passes(calls, missed, vpc, vpc0):
    return (calls <= lp.MAX_CALLS_PCT and missed <= lp.MAX_MISSED_PCT
            and np.isfinite(vpc) and vpc >= lp.MIN_VPC_RATIO * vpc0)


def summarize(S, grp, rng):
    idx, wt = lp.group_index(S, grp)
    if len(idx) == 0:
        return None
    fire = frozen_fire(S, idx, grp)
    base = S["base"][idx]
    pol_arr = lp.stats_matrix(S, idx, wt, fire.astype(float), np.where(fire, base, 0.0))
    alw_arr = lp.stats_matrix(S, idx, wt, np.ones(len(idx)), base)
    est = numbers(pol_arr, alw_arr)
    nm = S["nm"]
    boots, ok = [], []
    for _ in range(lp.N_BOOT):
        ix = rng.integers(0, nm, nm)
        b = numbers(pol_arr[ix], alw_arr[ix])
        boots.append(b[:2])
        ok.append(passes(*b))
    boots = np.array(boots, dtype=float)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return dict(group=grp, calls=est[0], calls_lo=lo[0], calls_hi=hi[0],
                missed=est[1], missed_lo=lo[1], missed_hi=hi[1],
                random_missed=100 - est[0], value_per_call=est[2], always_value_per_call=est[3],
                pass_share=100 * float(np.mean(ok)), examples=int(len(idx)))


def main():
    rng = np.random.default_rng(lp.SEED)
    levels, values = lp.load_pace_quantiles(lp.PACE)
    shape = lp.load_gap_shape()
    with open(lp.CHOICE) as f:
        choice = json.load(f)
    df = pd.read_csv(lp.RESULTS)
    df = df[df["system"] == "qwen"].copy()
    df["latency_top1_ms"] = pd.to_numeric(df["latency_top1_ms"], errors="coerce")
    df["semantic_sim"] = pd.to_numeric(df.get("semantic_sim"), errors="coerce")
    print(f"Source: {lp.RESULTS}; {df['message_id'].nunique()} messages, {len(df)} Qwen examples; "
          f"{lp.REPEATS} simulated typists per example; seed {lp.SEED}")
    print(f"Frozen policy: sentence at >= {pol.SENTENCE_MIN_WORDS} words typed; "
          f"word level at word starts and from {pol.MIN_LETTERS} letters in a word")
    S = lp.build_samples(df, choice, shape, levels, values, rng)
    print(f"All messages, no dev/test split (nothing is tuned). Intervals: 95 % bootstrap over messages, "
          f"{lp.N_BOOT} resamples\n")
    rows = []
    boot_rng = np.random.default_rng(lp.SEED + 3)
    print(f"{'group':<16}{'calls %':>18}{'missed %':>18}{'random missed':>15}{'value/call':>12}{'always':>8}{'meets rule':>12}")
    for grp in GROUPS:
        r = summarize(S, grp, boot_rng)
        if r is None:
            continue
        rows.append(r)
        print(f"{grp:<16}{r['calls']:>9.1f} [{r['calls_lo']:.0f}-{r['calls_hi']:.0f}]"
              f"{r['missed']:>10.1f} [{r['missed_lo']:.0f}-{r['missed_hi']:.0f}]"
              f"{r['random_missed']:>15.1f}{r['value_per_call']:>12.2f}{r['always_value_per_call']:>8.2f}"
              f"{r['pass_share']:>11.0f} %")
    os.makedirs("results", exist_ok=True)
    name = os.path.splitext(os.path.basename(lp.RESULTS))[0]
    out = f"results/frozen_policy_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()