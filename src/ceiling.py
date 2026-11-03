"""Phase 2a, Stage A7b: how good could ANY policy be with the inputs we have now?

Why: the learned policy (A7) did not beat the best simple rule. That leaves two readings:
  (a) the inputs (timing, words typed, letters in the word) simply do not contain more signal, or
  (b) the classifier was too weak to find it.
This script separates the two without any classifier. It groups decision points by CONTEXT
(for example "3 letters into a word, 9 words typed") and uses the best possible rule for that
grouping: call at the contexts where a suggestion most often delivers value, skip the others.
That is the best any policy can do if it only knows the context (the "ceiling" for that context).

Two versions of the ceiling are reported:
  honest     the delivery rate of each context is measured on the DEV half of the messages and
             applied to the TEST half (a real, usable policy; never sees the test answers);
  in-sample  the rates are measured on the test half itself (an optimistic upper bound; for
             contexts with many cells it is inflated, because it partly memorises the examples).
Coarse contexts have few cells, so their two versions agree and the ceiling can be trusted.
Fine contexts show how much of an apparent gain is only memorisation (honest falls, in-sample rises).

Contexts (all known when the key is pressed):
  letters in word                      0..8+ letters typed in the current word
  words typed                          1..15+ words typed so far
  characters typed (10 bins)           prefix length, binned by dev-half deciles
  letters x words                      the two above together (words in steps of 2)
  letters x words x previous gap       plus the previous gap in 4 bins (timing)
If the ceiling of the richest context is no better than the best simple rule, no policy built on
these inputs can do clearly better, and the next gain has to come from NEW inputs (A7c).

It uses the same simulated typists, dev/test split and staleness rule as src/learned_policy.py
(same seeds, so the same samples) and the same baselines (best simple rule per call budget; the
rule family is picked on the test half, which favours the rules, so the comparison is conservative).

Run from the repo root (branch phase2a-policy-pilot):
    uv run python -m src.ceiling                      # full run
    REPEATS=3 N_BOOT=100 uv run python -m src.ceiling # quick check
Writes results/ceiling_summary.csv and results/ceiling_curves.csv
"""
import json
import os

import numpy as np
import pandas as pd

try:
    from src import learned_policy as lp
except ImportError:  # run as a plain script from inside src/
    import learned_policy as lp

SUMMARY = "results/ceiling_summary.csv"
CURVES = "results/ceiling_curves.csv"
PRIOR_K = 20.0                        # weight of the overall rate when a context has few examples
N_THRESH = 80                         # points per frontier
CONTEXTS = ["letters in word", "words typed", "characters typed",
            "letters x words", "letters x words x previous gap"]
MAX_LETTERS, MAX_WORDS = 8, 15


def col(X, name):
    return X[:, lp.ALL_FEATURES.index(name)]


def make_edges(X_dev):
    """Bin edges from the dev half only."""
    return dict(gap=np.quantile(col(X_dev, "log_prev"), [0.25, 0.5, 0.75]),
                chars=np.unique(np.quantile(col(X_dev, "log_len"), np.linspace(0.1, 0.9, 9))))


def context_ids(X, name, edges):
    kss = np.minimum(col(X, "kss"), MAX_LETTERS).astype(int)
    nw = np.minimum(col(X, "n_words"), MAX_WORDS).astype(int)
    if name == "letters in word":
        return kss
    if name == "words typed":
        return nw
    if name == "characters typed":
        return np.digitize(col(X, "log_len"), edges["chars"])
    lw = kss * 100 + nw // 2
    if name == "letters x words":
        return lw
    return lw * 10 + np.digitize(col(X, "log_prev"), edges["gap"])


def cell_scores(ids_fit, good_fit, w_fit, ids_apply, prior):
    """Smoothed delivery rate per context cell, fitted on one set of rows, read for another."""
    uniq, inv = np.unique(ids_fit, return_inverse=True)
    n = np.bincount(inv, weights=w_fit)
    g = np.bincount(inv, weights=w_fit * good_fit)
    rate = (g + PRIOR_K * prior) / (n + PRIOR_K)
    pos = np.minimum(np.searchsorted(uniq, ids_apply), len(uniq) - 1)
    known = uniq[pos] == ids_apply
    return np.where(known, rate[pos], prior), len(uniq)


def frontier(S, idx, wt, scores):
    """Call where the score is high: one point per threshold. Returns (nm, n_points, N_STAT)."""
    u = np.unique(scores)
    if len(u) > N_THRESH:
        u = np.unique(np.quantile(scores, np.linspace(0, 1, N_THRESH)))
    base = S["base"][idx]
    pts = []
    for th in [u.max() + 1.0] + list(u[::-1]):          # first point: never call
        inv = scores >= th
        pts.append(lp.stats_matrix(S, idx, wt, inv.astype(float), np.where(inv, base, 0.0)))
    return np.stack(pts, axis=1)


def rate_line(label, ids, wt, good, cap=None):
    out = []
    for v in np.unique(ids):
        m = ids == v
        out.append(f"{v}:{100 * (wt[m] * good[m]).sum() / wt[m].sum():.0f}")
    return f"  delivers (%) by {label}: " + "  ".join(out)


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
    S = lp.build_samples(df, choice, shape, levels, values, rng)
    nm = S["nm"]
    perm = np.random.default_rng(lp.SPLIT_SEED).permutation(nm)
    dev_ids, test_ids = np.sort(perm[: nm // 2]), np.sort(perm[nm // 2:])
    dev_rows, test_rows = np.isin(S["mi"], dev_ids), np.isin(S["mi"], test_ids)
    print(f"Messages split at random (seed {lp.SPLIT_SEED}): {len(dev_ids)} dev, {len(test_ids)} test "
          f"(same split as learned_policy.py)")
    print(f"Ceiling = best possible policy that only knows the context. honest: rates from dev, applied to test; "
          f"in-sample: rates from test (optimistic).")
    print(f"Intervals: 95 % bootstrap over test messages, {lp.N_BOOT} resamples, paired across policies\n")

    boot_rng = np.random.default_rng(lp.SEED + 2)
    summary, curve_rows = [], []
    budgets = lp.BUDGETS

    for grp in [lp.MIX] + lp.GRANS:
        idx, wt = lp.group_index(S, grp)
        dev_loc, test_loc = np.where(dev_rows[idx])[0], np.where(test_rows[idx])[0]
        if len(dev_loc) < 50 or len(test_loc) < 50:
            print(f"== {grp}: too few examples, skipped ==\n")
            continue
        X = S["feat"][idx]
        good = S["good"][idx].astype(float)
        idx_t, wt_t = idx[test_loc], wt[test_loc]
        edges = make_edges(X[dev_loc])
        prior_dev = float((wt[dev_loc] * good[dev_loc]).sum() / wt[dev_loc].sum())
        prior_test = float((wt_t * good[test_loc]).sum() / wt_t.sum())
        good_share = 100 * prior_test
        rand = {b: 100.0 - b for b in budgets}
        orc = {b: max(0.0, 100.0 * (1.0 - b / good_share)) for b in budgets}

        # baselines: simple rules on the test half
        fams = lp.rule_curves(S, idx_t, wt_t)
        always = lp.stats_matrix(S, idx_t, wt_t, np.ones(len(idx_t)), S["base"][idx_t]).sum(0)

        # ceilings per context
        arrs, ncells = {}, {}
        for c in CONTEXTS:
            ids = context_ids(X, c, edges)
            s_h, nc = cell_scores(ids[dev_loc], good[dev_loc], wt[dev_loc], ids[test_loc], prior_dev)
            s_i, _ = cell_scores(ids[test_loc], good[test_loc], wt_t, ids[test_loc], prior_test)
            arrs[f"{c} (honest)"] = frontier(S, idx_t, wt_t, s_h)
            arrs[f"{c} (in-sample)"] = frontier(S, idx_t, wt_t, s_i)
            ncells[c] = nc
        point = {k: a.sum(0) for k, a in arrs.items()}
        rule_point = {f: a.sum(0) for f, a in fams.items()}
        best_rule = {}
        for b in budgets:
            cand = [(lp.read_out(rule_point[f], b), f) for f in lp.RULE_FAMILIES]
            cand = [(m, f) for m, f in cand if np.isfinite(m)]
            best_rule[b] = min(cand) if cand else (np.nan, None)

        # paired bootstrap over test messages
        keys = list(arrs)
        rule_used = sorted({f for _, f in best_rule.values() if f})
        boot = {k: np.zeros((lp.N_BOOT, len(budgets))) for k in keys + rule_used}
        for i in range(lp.N_BOOT):
            ix = boot_rng.choice(test_ids, len(test_ids))
            for k in keys:
                tot = arrs[k][ix].sum(0)
                for bi, b in enumerate(budgets):
                    boot[k][i, bi] = lp.read_out(tot, b)
            for f in rule_used:
                tot = fams[f][ix].sum(0)
                for bi, b in enumerate(budgets):
                    boot[f][i, bi] = lp.read_out(tot, b)

        print(f"== {grp}: always-invoke delivers at {good_share:.1f} % of test decision points "
              f"(the oracle needs that few calls) ==")
        print("How often a call delivers, by context (all messages):")
        print(rate_line("words typed", np.minimum(col(X, "n_words"), MAX_WORDS).astype(int), wt, good))
        print(rate_line("letters in word", np.minimum(col(X, "kss"), MAX_LETTERS).astype(int), wt, good))
        print(f"\nTEST half, missed % at each call budget [95 % interval]:")
        print(f"{'policy':<52}{'cells':>6}" + "".join(f"{'@' + str(b) + '% calls':>20}" for b in budgets))
        print(f"{'random skip':<52}{'':>6}" + "".join(f"{rand[b]:>20.1f}" for b in budgets))
        print(f"{'oracle (knows the answer)':<52}{'':>6}" + "".join(f"{orc[b]:>20.1f}" for b in budgets))
        cells = []
        for bi, b in enumerate(budgets):
            m, f = best_rule[b]
            lo, hi = lp.pct(boot[f][:, bi]) if f else (np.nan, np.nan)
            cells.append(lp.fmt(m, lo, hi))
            summary.append(dict(group=grp, policy="best simple rule", detail=f, budget=b, missed=m, lo=lo, hi=hi))
        print(f"{'best simple rule (family picked per budget)':<52}{'':>6}" + "".join(f"{c:>20}" for c in cells))
        print("  (best simple rule at each budget: " +
              ", ".join(f"{b}% -> {best_rule[b][1]}" for b in budgets) + ")")
        for k in keys:
            c = k.rsplit(" (", 1)[0]
            row = []
            for bi, b in enumerate(budgets):
                est = lp.read_out(point[k], b)
                lo, hi = lp.pct(boot[k][:, bi])
                row.append(lp.fmt(est, lo, hi))
                summary.append(dict(group=grp, policy=k, detail=str(ncells[c]), budget=b, missed=est, lo=lo, hi=hi))
            print(f"{'ceiling: ' + k:<52}{ncells[c]:>6}" + "".join(f"{x:>20}" for x in row))
            calls, missed = lp.curve_xy(point[k])
            for cx, my in zip(calls, missed):
                curve_rows.append(dict(group=grp, policy=k, calls=cx, missed=my))
        for f in lp.RULE_FAMILIES:
            calls, missed = lp.curve_xy(rule_point[f])
            for cx, my in zip(calls, missed):
                curve_rows.append(dict(group=grp, policy=f"rule: {f}", calls=cx, missed=my))

        print("\nHonest ceiling minus best simple rule, in missed points (negative = the ceiling is better) "
              "[95 % interval of the difference]:")
        for c in CONTEXTS:
            k = f"{c} (honest)"
            parts = []
            for bi, b in enumerate(budgets):
                m, f = best_rule[b]
                est = lp.read_out(point[k], b)
                if not (np.isfinite(est) and f):
                    parts.append(f"@{b}%: n/a")
                    continue
                lo, hi = lp.pct(boot[k][:, bi] - boot[f][:, bi])
                parts.append(f"@{b}%: {est - m:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            print(f"  {c:<34}" + "   ".join(parts))

        print(f"\nProposed selection rule (calls <= {lp.MAX_CALLS_PCT:.0f} %, missed <= {lp.MAX_MISSED_PCT:.0f} %, "
              f"value/call >= {lp.MIN_VPC_RATIO} x always), fewest calls that pass, point estimate:")
        res = [(f"rule: {f}", lp.pass_calls(rule_point[f], always)) for f in lp.RULE_FAMILIES]
        res += [(f"ceiling: {k}", lp.pass_calls(point[k], always)) for k in keys]
        for name, v in res:
            print(f"  {name:<58}" + (f"{v:.0f} % calls" if np.isfinite(v) else "never at <= 50 % calls"))
            summary.append(dict(group=grp, policy=name, detail="pass_calls", budget="pass", missed=v,
                                lo=np.nan, hi=np.nan))
        print()

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    pd.DataFrame(curve_rows).to_csv(CURVES, index=False)
    print(f"Saved {SUMMARY} and {CURVES}")


if __name__ == "__main__":
    main()