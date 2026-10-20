"""Phase 2a, Stage A7: learned invocation policy, evaluated against the simple rules.

What this does (same simulation and staleness rule as policy_eval_v3.py):
  1. Simulate typists (AmazonQAC pace x Aalto gap shape) at the cut point of every Qwen example
     in phase1_results_v3.csv and record, for each simulated typist, what each policy would do.
  2. Train a classifier that predicts, from information available when the key is pressed,
     whether a call here would DELIVER value on arrival (label includes staleness).
  3. Evaluate it out-of-sample: 5-fold cross-validation grouped by MESSAGE, so no message is
     ever in training and test at the same time.
  4. Compare on the whole trade-off curve (calls vs missed) instead of one pass/fail line:
        random skip   = floor   (at c % of calls it misses (100 - c) %)
        oracle        = ceiling (calls only where the call would deliver)
        pause T, boundary + mid-word T = the best simple rules (curves over T)
        learned policies (logistic regression, small gradient-boosted trees)
     and report "share of the random-to-oracle gap closed", with 95 % intervals from a
     bootstrap over messages (paired, so differences between policies have their own interval).

Calls = share of decision points at which the policy calls the model (fire@dec). Missed and
calls are measured at the SAME position, which removes the mismatch described in plan 4.3.

Features (all known when the key is pressed; nothing from the future):
  timing   : previous gap, mean of the last 5 gaps, typist's mean gap so far, previous/mean
  position : characters since the last space, last key is a space, prefix length, words typed,
             last key is a letter / digit
A learned policy may also require a short silence first (gate tau, mid-word only, as in
"boundary + mid-word T"): it is asked only if the user has been silent for tau seconds, and the
call starts then.

Run from the repo root (branch phase2a-policy-pilot):
    uv run python -m src.learned_policy            # full run
    REPEATS=3 uv run python -m src.learned_policy  # quick check
Writes results/learned_policy_summary.csv and results/learned_policy_curves.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from src.metrics import keystrokes_saved
    from src.typing_sim import load_pace_quantiles, sample_pace
    from src.gap_shape import load_gap_shape, simulate_typing_empirical
except ImportError:  # run as a plain script from inside src/
    from metrics import keystrokes_saved
    from typing_sim import load_pace_quantiles, sample_pace
    from gap_shape import load_gap_shape, simulate_typing_empirical

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = os.environ.get("RESULTS", "phase1_results_v3.csv")
CHOICE = "metric_choice.json"
PACE = "amazonqac_pace_quantiles.csv"
SUMMARY = "results/learned_policy_summary.csv"
CURVES = "results/learned_policy_curves.csv"
SEED = 42
REPEATS = int(os.environ.get("REPEATS", 10))
N_BOOT = int(os.environ.get("N_BOOT", 1000))
N_FOLDS = 5
GRANS = ["partial_word", "next_word", "phrase", "sentence"]
MIX = "word-level mix"
GOOD_SIM = 0.4                       # same provisional threshold as policy_eval_v3.py
BUDGETS = (20, 35, 50)               # call budgets (% of decision points) for the read-out
RULE_T = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
GATES = (0.0, 0.4)                   # silence a learned policy may require mid-word (s)
QUANTILES = np.linspace(0.02, 0.98, 25)   # threshold sweep: share of training scores to pass
# Pre-specified primary learned policy (chosen before looking at results): the simplest model with
# the richest inputs, with the mid-word silence gate that the combined rule already showed helps.
# All other learned variants are reported as ablations only.
PRIMARY = "logreg [timing+position] gate 0.4s"
N_STAT = 6                           # weight, calls, good, missed, value, base value
FALLBACK_GAP = 0.67                  # AmazonQAC median pace, used when there is no history yet

TIMING = ["log_prev", "log_mean5", "log_speed", "log_ratio"]
POSITION = ["kss", "at_space", "log_len", "n_words", "is_alpha", "is_digit"]
FEATURE_SETS = {"timing": TIMING, "timing+position": TIMING + POSITION}
MODELS = {
    "logreg [timing]": ("logreg", "timing"),
    "logreg [timing+position]": ("logreg", "timing+position"),
    "gbt [timing+position]": ("gbt", "timing+position"),
}


def chosen_metric(choice, g):
    entry = choice.get(g, {})
    names = {"keystrokes": "keystrokes_saved", "embeddings": "semantic_similarity"}
    if isinstance(entry, dict) and entry.get("metric") in names:
        return names[entry["metric"]]
    return "semantic_similarity" if g == "sentence" else "keystrokes_saved"


def features_at(gaps, full, dec):
    if dec >= 1:
        prev = gaps[dec - 1]
        mean5 = gaps[max(0, dec - 5):dec].mean()
        speed = gaps[:dec].mean()
    else:
        prev = mean5 = speed = FALLBACK_GAP
    prev, mean5, speed = max(prev, 1e-3), max(mean5, 1e-3), max(speed, 1e-3)
    ch = full[dec]
    head = full[:dec + 1]
    return dict(
        log_prev=np.log(prev), log_mean5=np.log(mean5), log_speed=np.log(speed),
        log_ratio=np.log(prev / speed),
        kss=len(head.split(" ")[-1]),
        at_space=float(ch == " "), log_len=np.log1p(dec + 1), n_words=head.count(" ") + 1,
        is_alpha=float(ch.isalpha()), is_digit=float(ch.isdigit()))


def build_samples(df, choice, shape, levels, values, rng):
    """One row per (example, simulated typist). Returns dict of arrays."""
    ratio = float((df["prefix"].str.len() / df["prefix_words"].clip(lower=1)).mean()) - 1.0
    wmix = {"partial_word": ratio, "next_word": 1.0}
    rule_names = [f"pause {t}s" for t in RULE_T] + [f"boundary + mid-word {t}s" for t in RULE_T]
    rec = dict(mi=[], gran=[], wmix=[], feat=[], good=[], base=[],
               rule_val=[], rule_inv=[], gate_ok=[], gate_val=[])
    msg_ids = sorted(df["message_id"].unique())
    midx = {m: i for i, m in enumerate(msg_ids)}
    used = {g: 0 for g in GRANS}
    for g in GRANS:
        sub = df[df["granularity"] == g]
        use_ks = chosen_metric(choice, g) == "keystrokes_saved"
        for row in sub.itertuples():
            prefix = str(row.prefix) if pd.notna(row.prefix) else ""
            ref = str(row.reference) if pd.notna(row.reference) else ""
            pred = str(row.prediction) if pd.notna(row.prediction) else ""
            lat = row.latency_top1_ms
            if not prefix or not ref or pd.isna(lat):
                continue
            sim = row.semantic_sim
            if not use_ks and pd.isna(sim):
                continue
            L = lat / 1000
            ks0 = keystrokes_saved(pred, ref)
            if g == "partial_word":
                full, dec = prefix + ref, len(prefix) - 1
            else:
                full, dec = prefix + " " + ref, len(prefix)
            used[g] += 1
            is_space = np.array([c == " " for c in full[:-1]])

            def value_at(times, wait):
                arrival = times[dec] + wait + L
                k = int(np.searchsorted(times[dec + 1:], arrival, side="right"))
                if ks0 >= k:
                    return float(ks0 - k) if use_ks else float(sim)
                return 0.0

            for _ in range(REPEATS):
                pace = sample_pace(rng, levels, values)
                tl = simulate_typing_empirical(full, pace, shape, rng)
                times = np.array([t for t, _ in tl])
                gaps = np.diff(times)
                gd, sp = gaps[dec], bool(is_space[dec])
                feat = features_at(gaps, full, dec)
                rv, ri = [], []
                for t in RULE_T:                                   # pause T
                    inv = gd > t
                    ri.append(inv)
                    rv.append(value_at(times, t) if inv else 0.0)
                for t in RULE_T:                                   # boundary + mid-word T
                    inv = sp or gd > t
                    ri.append(inv)
                    rv.append(value_at(times, 0.0 if sp else t) if inv else 0.0)
                always_v = value_at(times, 0.0)
                gok, gval = [], []
                for tau in GATES:                                  # learned-policy gates
                    wait = 0.0 if sp else tau
                    gok.append(sp or gd > tau)
                    gval.append(value_at(times, wait))
                rec["mi"].append(midx[row.message_id])
                rec["gran"].append(g)
                rec["wmix"].append(wmix.get(g, np.nan))
                rec["feat"].append([feat[c] for c in FEATURE_SETS["timing+position"]])
                rec["good"].append(always_v > 0 if use_ks else always_v >= GOOD_SIM)
                rec["base"].append(always_v)
                rec["rule_val"].append(rv)
                rec["rule_inv"].append(ri)
                rec["gate_ok"].append(gok)
                rec["gate_val"].append(gval)
    out = {k: np.array(v) for k, v in rec.items()}
    out["use_ks"] = {g: chosen_metric(choice, g) == "keystrokes_saved" for g in GRANS}
    out["rule_names"] = rule_names
    out["nm"] = len(msg_ids)
    out["used"] = used
    out["ratio"] = ratio
    return out


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=100,
                                          min_samples_leaf=50, random_state=SEED)


def label_of(val, use_ks):
    return (val > 0) if use_ks else (val >= GOOD_SIM)


def group_index(S, grp):
    if grp == MIX:
        sel = np.isin(S["gran"], ["partial_word", "next_word"])
        return np.where(sel)[0], S["wmix"][sel]
    sel = S["gran"] == grp
    return np.where(sel)[0], np.ones(sel.sum())


def stats_matrix(S, idx, wt, invoked, value):
    """Per-message sums [weight, calls, good, missed, value, base] for one policy."""
    nm = S["nm"]
    good = S["good"][idx]
    out = np.zeros((nm, N_STAT))
    mi = S["mi"][idx]
    vals = [wt, wt * invoked, wt * good, wt * (good & (value <= 0)), wt * value, wt * S["base"][idx]]
    for j, v in enumerate(vals):
        out[:, j] = np.bincount(mi, weights=v, minlength=nm)
    return out


def rule_curves(S, idx, wt):
    """{family: (params, array (nm, n_params, N_STAT))} for the simple rules."""
    fam = {"pause": [], "boundary + mid-word": []}
    for j, name in enumerate(S["rule_names"]):
        f = "pause" if name.startswith("pause") else "boundary + mid-word"
        fam[f].append(stats_matrix(S, idx, wt, S["rule_inv"][idx, j].astype(float), S["rule_val"][idx, j]))
    return {f: np.stack(v, axis=1) for f, v in fam.items()}


def learned_curves(S, grp, idx, wt):
    """Out-of-fold curves for each (model, gate): {name: array (nm, n_quantiles, N_STAT)}."""
    nm = S["nm"]
    use_ks = S["use_ks"]["partial_word" if grp == MIX else grp]
    X_all = S["feat"][idx]
    mi = S["mi"][idx]
    gran = S["gran"][idx]
    # sentence similarity labels for the mix are keystroke based (mix only pools keystroke groups)
    curves = {}
    cols_all = FEATURE_SETS["timing+position"]
    folds = list(GroupKFold(n_splits=min(N_FOLDS, len(np.unique(mi)))).split(X_all, groups=mi))
    for mname, (kind, fset) in MODELS.items():
        cols = [cols_all.index(c) for c in FEATURE_SETS[fset]]
        for gi, tau in enumerate(GATES):
            ok = S["gate_ok"][idx, gi]
            val = S["gate_val"][idx, gi]
            y = label_of(val, use_ks)
            acc = np.zeros((nm, len(QUANTILES), N_STAT))
            for tr, te in folds:
                trm = tr[ok[tr]]
                if len(trm) < 20 or y[trm].min() == y[trm].max():
                    continue
                model = make_model(kind)
                fit_kw = {}
                sw = wt[trm]
                if kind == "logreg":
                    fit_kw = {"logisticregression__sample_weight": sw}
                else:
                    fit_kw = {"sample_weight": sw}
                model.fit(X_all[trm][:, cols], y[trm], **fit_kw)
                s_tr = model.predict_proba(X_all[trm][:, cols])[:, 1]
                s_te = np.full(len(te), -1.0)
                tem = ok[te]
                if tem.any():
                    s_te[tem] = model.predict_proba(X_all[te][tem][:, cols])[:, 1]
                for qi, q in enumerate(QUANTILES):
                    theta = np.quantile(s_tr, 1.0 - q)       # pass the top q share of training scores
                    fired = (s_te >= theta) & tem
                    value = np.where(fired, val[te], 0.0)
                    acc[:, qi, :] += stats_rows(S, idx[te], wt[te], fired.astype(float), value)
            curves[f"{mname} gate {tau:g}s"] = acc
    return curves


def stats_rows(S, rows, wt, invoked, value):
    nm = S["nm"]
    good = S["good"][rows]
    mi = S["mi"][rows]
    out = np.zeros((nm, N_STAT))
    vals = [wt, wt * invoked, wt * good, wt * (good & (value <= 0)), wt * value, wt * S["base"][rows]]
    for j, v in enumerate(vals):
        out[:, j] = np.bincount(mi, weights=v, minlength=nm)
    return out


def curve_xy(arr):
    """arr (n_points, N_STAT) summed over messages -> calls %, missed % sorted by calls."""
    with np.errstate(divide="ignore", invalid="ignore"):
        calls = 100 * arr[:, 1] / arr[:, 0]
        missed = 100 * arr[:, 3] / arr[:, 2]
    o = np.argsort(calls)
    return calls[o], missed[o]


def pct(x, qs=(2.5, 97.5)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) == 0 else tuple(np.percentile(x, qs))


def read_out(arr, budget):
    calls, missed = curve_xy(arr)
    if budget < calls.min() - 1e-9 or budget > calls.max() + 1e-9:
        return np.nan
    return float(np.interp(budget, calls, missed))


def main():
    rng = np.random.default_rng(SEED)
    levels, values = load_pace_quantiles(PACE)
    shape = load_gap_shape()
    with open(CHOICE) as f:
        choice = json.load(f)
    df = pd.read_csv(RESULTS)
    df = df[df["system"] == "qwen"].copy()
    df["latency_top1_ms"] = pd.to_numeric(df["latency_top1_ms"], errors="coerce")
    df["semantic_sim"] = pd.to_numeric(df.get("semantic_sim"), errors="coerce")
    print(f"Source: {RESULTS}; {df['message_id'].nunique()} messages, {len(df)} Qwen examples; "
          f"{REPEATS} simulated typists per example; seed {SEED}")
    print("Typing model: Aalto gap shape x AmazonQAC pace; latency = single-suggestion time")
    S = build_samples(df, choice, shape, levels, values, rng)
    print(f"Samples: {len(S['mi'])} (examples used: {S['used']}); mix weight mid-word x {S['ratio']:.2f}")
    print(f"Evaluation: {N_FOLDS}-fold CV grouped by message; bootstrap {N_BOOT} resamples over messages\n")

    summary, curve_rows = [], []
    boot_rng = np.random.default_rng(SEED + 1)
    nm = S["nm"]
    for grp in [MIX] + GRANS:
        idx, wt = group_index(S, grp)
        if len(idx) == 0:
            continue
        fams = {}
        for f, arr in rule_curves(S, idx, wt).items():
            fams[f] = arr
        for name, arr in learned_curves(S, grp, idx, wt).items():
            fams[name] = arr
        # oracle (calls only where a call delivers value) from the always-invoke outcome
        good_share = 100 * float((wt * S["good"][idx]).sum() / wt.sum())
        point = {f: a.sum(0) for f, a in fams.items()}
        for f, a in point.items():
            for k in range(a.shape[0]):
                c, m = 100 * a[k, 1] / a[k, 0], 100 * a[k, 3] / a[k, 2]
                curve_rows.append(dict(group=grp, family=f, point=k, calls=c, missed=m))
        # bootstrap over messages, same resample for every family (paired)
        boots = {f: np.zeros((N_BOOT, len(BUDGETS))) for f in fams}
        for b in range(N_BOOT):
            ix = boot_rng.integers(0, nm, nm)
            for f, a in fams.items():
                tot = a[ix].sum(0)
                for bi, bud in enumerate(BUDGETS):
                    boots[f][b, bi] = read_out(tot, bud)
        print(f"== {grp}: always-invoke delivers at {good_share:.1f} % of decision points "
              f"(oracle needs that few calls) ==")
        print(f"{'policy':<40}" + "".join(f"{'missed @' + str(b) + '% calls':>22}" for b in BUDGETS))
        rows_print = []
        for f in fams:
            cells = []
            for bi, bud in enumerate(BUDGETS):
                est = read_out(point[f], bud)
                lo, hi = pct(boots[f][:, bi])
                cells.append((est, lo, hi))
                summary.append(dict(group=grp, family=f, budget=bud, missed=est, lo=lo, hi=hi,
                                    gap_closed=np.nan))
            rows_print.append((f, cells))
        rand = {b: 100.0 - b for b in BUDGETS}
        orc = {b: max(0.0, 100.0 * (1.0 - b / good_share)) for b in BUDGETS}
        print(f"{'random skip':<40}" + "".join(f"{rand[b]:>22.1f}" for b in BUDGETS))
        print(f"{'oracle':<40}" + "".join(f"{orc[b]:>22.1f}" for b in BUDGETS))
        for f, cells in rows_print:
            line = f"{f:<40}"
            for (est, lo, hi) in cells:
                line += f"{'n/a':>22}" if np.isnan(est) else f"{f'{est:.1f} [{lo:.0f}-{hi:.0f}]':>22}"
            print(line)
        # gap closed: share of the random-to-oracle distance covered.
        # Primary learned policy (pre-specified) vs the BEST simple rule at each budget
        # (the baseline is allowed to pick its best family, the learned policy is not).
        rule_names = [f for f in fams if f in ("pause", "boundary + mid-word")]
        print(f"-- primary learned policy ({PRIMARY}) vs best simple rule; share of the random-to-oracle gap closed --")
        for bi, bud in enumerate(BUDGETS):
            denom = rand[bud] - orc[bud]
            ml = read_out(point[PRIMARY], bud)
            cand = [(read_out(point[f], bud), f) for f in rule_names]
            cand = [(m, f) for m, f in cand if np.isfinite(m)]
            if not np.isfinite(ml) or not cand or denom <= 0:
                print(f"  {bud}% calls: not available (a curve does not reach this call budget)")
                continue
            mr, best_r = min(cand)
            diff = boots[PRIMARY][:, bi] - boots[best_r][:, bi]       # negative = learned misses less
            lo, hi = pct(diff)
            print(f"  {bud}% calls: learned {100 * (rand[bud] - ml) / denom:.0f} % | best simple rule ({best_r}) "
                  f"{100 * (rand[bud] - mr) / denom:.0f} % | learned minus rule in missed points: "
                  f"{ml - mr:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        print()
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    pd.DataFrame(curve_rows).to_csv(CURVES, index=False)
    print(f"Saved {SUMMARY} and {CURVES}")


if __name__ == "__main__":
    main()
