"""Phase 2a, Stage A7: learned invocation policy, evaluated against the simple rules.

What this does (same simulation and staleness rule as policy_eval_v3.py):
  1. Simulate typists (AmazonQAC pace x Aalto gap shape) at the cut point of every Qwen example
     in phase1_results_v3.csv and record, for each simulated typist, what each policy would do.
  2. Train a classifier that predicts, from information available when the key is pressed,
     whether a call here would DELIVER value on arrival (label includes staleness).
  3. Keep selection and final scoring apart: the messages are split once, at random, into a
     DEV half and a TEST half.
        - the learned variant (model, features, wait) is CHOSEN on the dev half by 5-fold
          cross-validation grouped by message;
        - the chosen variant is trained on the whole dev half and scored on the test half,
          which was never used for any choice. Every number in the main tables is test-half.
  4. Compare on the whole trade-off curve (calls vs missed), against
        random skip = floor, oracle = ceiling, the simple rules (pause T, boundary + mid-word T),
     and report the share of the random-to-oracle gap closed with 95 % bootstrap intervals over
     test messages (paired, so a difference between two policies has its own interval).
  5. Apply the proposed selection rule (calls <= 50 % of always-invoke, missed <= 25 %,
     value per call >= 0.95 x always-invoke's): how often does each policy pass in the bootstrap,
     and at how few calls?
  6. Simple position/content rules (letters typed in the word, words typed so far) are part of the
     baseline, next to the timing rules, so "learned beats simple rules" is a fair comparison.
  7. Feature check: models using only prefix length, only word shape, or everything except length,
     and the standardised coefficients of the full model, to show what carries the gain.

Calls = share of decision points at which the policy calls the model (fire@dec). Missed and
calls are measured at the SAME position, which removes the mismatch described in plan 4.3.

Features (all known when the key is pressed; nothing from the future):
  timing   : previous gap, mean of the last 5 gaps, typist's mean gap so far, previous/mean
  position : characters since the last space, last key is a space, prefix length, words typed,
             last key is a letter / digit
A learned variant may also require a short silence first (gate tau, mid-word only, as in
"boundary + mid-word T"): it is asked only if the user has been silent for tau seconds, and the
call starts then.

Run from the repo root (branch phase2a-policy-pilot):
    uv run python -m src.learned_policy                          # full run
    REPEATS=3 N_BOOT=100 uv run python -m src.learned_policy     # quick check
    SPLIT_SEED=8 uv run python -m src.learned_policy             # a different dev/test split
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
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 7))   # dev/test split of the messages
MAX_CALLS_PCT, MAX_MISSED_PCT, MIN_VPC_RATIO = 50.0, 25.0, 0.95   # proposed selection rule
N_STAT = 6                           # weight, calls, good, missed, value, base value
FALLBACK_GAP = 0.67                  # AmazonQAC median pace, used when there is no history yet

TIMING = ["log_prev", "log_mean5", "log_speed", "log_ratio"]
POSITION = ["kss", "at_space", "log_len", "n_words", "is_alpha", "is_digit"]
ALL_FEATURES = TIMING + POSITION
FEATURE_SETS = {
    "timing": TIMING,
    "timing+position": TIMING + POSITION,
    # feature check (diagnostics only, never selected):
    "prefix length only": ["log_len", "n_words"],
    "word shape only": ["kss", "at_space", "is_alpha", "is_digit"],
    "word shape + length": ["kss", "at_space", "is_alpha", "is_digit", "log_len", "n_words"],
    "timing+position, no length": TIMING + ["kss", "at_space", "is_alpha", "is_digit"],
}
MODELS = {                          # candidates the dev half chooses from (each with each gate)
    "logreg [timing]": ("logreg", "timing"),
    "logreg [timing+position]": ("logreg", "timing+position"),
    "gbt [timing+position]": ("gbt", "timing+position"),
    "logreg [word shape]": ("logreg", "word shape only"),
    "logreg [prefix length]": ("logreg", "prefix length only"),
    "logreg [word shape+length]": ("logreg", "word shape + length"),
}
ABLATIONS = {                       # diagnostics only, no wait, scored on the test half
    "logreg [timing+position, no length]": ("logreg", "timing+position, no length"),
}

# Simple (not learned) position and content rules, added after the first run showed that the
# earlier simple rules were timing rules only. Each is a one-parameter family; fires at key press.
#   kss = letters typed in the current word, sp = last key is a space, nw = words typed so far
POS_RULES = {
    "min letters in word": (range(1, 9), lambda kss, sp, nw, n: kss >= n),
    "word start + min letters": (range(1, 9), lambda kss, sp, nw, n: sp | (kss >= n)),
    "min words typed": (range(1, 13), lambda kss, sp, nw, n: nw >= n),
    "word start + min words": (range(1, 13), lambda kss, sp, nw, n: sp & (nw >= n)),
}
RULE_FAMILIES = ["pause", "boundary + mid-word"] + list(POS_RULES)
RULE_PARAMS = {"pause": list(RULE_T), "boundary + mid-word": list(RULE_T),
               **{f: list(r) for f, (r, _) in POS_RULES.items()}}


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
                rec["feat"].append([feat[c] for c in ALL_FEATURES])
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
    out = {f: np.stack(v, axis=1) for f, v in fam.items()}
    X = S["feat"][idx]
    kss, sp, nw = (X[:, ALL_FEATURES.index(c)] for c in ("kss", "at_space", "n_words"))
    sp = sp > 0.5
    base = S["base"][idx]
    for f, (params, rule) in POS_RULES.items():
        pts = []
        for n in params:
            inv = rule(kss, sp, nw, n)
            pts.append(stats_matrix(S, idx, wt, inv.astype(float), np.where(inv, base, 0.0)))
        out[f] = np.stack(pts, axis=1)
    return out


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


def curve_xyv(arr):
    """arr (n_points, N_STAT) -> calls %, missed %, value per call, sorted by calls."""
    with np.errstate(divide="ignore", invalid="ignore"):
        calls = 100 * arr[:, 1] / arr[:, 0]
        missed = 100 * arr[:, 3] / arr[:, 2]
        vpc = np.where(arr[:, 1] > 0, arr[:, 4] / arr[:, 1], np.nan)
    o = np.argsort(calls)
    return calls[o], missed[o], vpc[o]


def pass_calls(tot, always_tot):
    """Smallest call share (1..50 %) at which the policy meets the proposed selection rule on
    the interpolated curve; NaN if it never does. 'always' is the reference for value per call."""
    calls, missed, vpc = curve_xyv(tot)
    grid = np.arange(1.0, MAX_CALLS_PCT + 0.5, 1.0)
    vpc0 = always_tot[4] / always_tot[1]
    inside = (grid >= calls.min() - 1e-9) & (grid <= calls.max() + 1e-9)
    m = np.interp(grid, calls, missed)
    v = np.interp(grid, calls, np.nan_to_num(vpc, nan=0.0))
    ok = inside & (m <= MAX_MISSED_PCT) & (v >= MIN_VPC_RATIO * vpc0)
    return float(grid[ok].min()) if ok.any() else np.nan


def run_variant(S, idx, wt, use_ks, tr, te, kind, fset, gi):
    """Train on rows `tr`, score rows `te` (both local indices into idx).
    Returns (per-message stats over the threshold sweep for the test rows, fitted model, cols)."""
    nm = S["nm"]
    cols = [ALL_FEATURES.index(c) for c in FEATURE_SETS[fset]]
    X = S["feat"][idx]
    ok = S["gate_ok"][idx, gi]
    val = S["gate_val"][idx, gi]
    y = label_of(val, use_ks)
    acc = np.zeros((nm, len(QUANTILES), N_STAT))
    trm = tr[ok[tr]]
    if len(trm) < 20 or y[trm].min() == y[trm].max():
        return acc, None, cols
    model = make_model(kind)
    sw = wt[trm]
    model.fit(X[trm][:, cols], y[trm],
              **({"logisticregression__sample_weight": sw} if kind == "logreg" else {"sample_weight": sw}))
    s_tr = model.predict_proba(X[trm][:, cols])[:, 1]
    tem = ok[te]
    s_te = np.full(len(te), -1.0)
    if tem.any():
        s_te[tem] = model.predict_proba(X[te][tem][:, cols])[:, 1]
    for qi, q in enumerate(QUANTILES):
        theta = np.quantile(s_tr, 1.0 - q)               # pass the top q share of training scores
        fired = (s_te >= theta) & tem
        acc[:, qi, :] += stats_rows(S, idx[te], wt[te], fired.astype(float), np.where(fired, val[te], 0.0))
    return acc, model, cols


def dev_cv_score(S, idx, wt, use_ks, dev_loc, kind, fset, gi):
    """Mean missed % over the call budgets, by grouped CV inside the dev half only."""
    mi = S["mi"][idx][dev_loc]
    n_splits = min(N_FOLDS, len(np.unique(mi)))
    acc = np.zeros((S["nm"], len(QUANTILES), N_STAT))
    for tr, te in GroupKFold(n_splits=n_splits).split(dev_loc, groups=mi):
        a, _, _ = run_variant(S, idx, wt, use_ks, dev_loc[tr], dev_loc[te], kind, fset, gi)
        acc += a
    tot = acc.sum(0)
    vals = [read_out(tot, b) for b in BUDGETS]
    return float(np.mean([v if np.isfinite(v) else 100.0 for v in vals]))


def fmt(est, lo, hi):
    return "n/a" if not np.isfinite(est) else f"{est:.1f} [{lo:.0f}-{hi:.0f}]"


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
    nm = S["nm"]
    perm = np.random.default_rng(SPLIT_SEED).permutation(nm)
    dev_ids, test_ids = np.sort(perm[: nm // 2]), np.sort(perm[nm // 2:])
    dev_rows, test_rows = np.isin(S["mi"], dev_ids), np.isin(S["mi"], test_ids)
    print(f"Samples: {len(S['mi'])} (examples used: {S['used']}); mix weight mid-word x {S['ratio']:.2f}")
    print(f"Messages split at random (seed {SPLIT_SEED}): {len(dev_ids)} dev (variant chosen here, "
          f"{N_FOLDS}-fold CV by message), {len(test_ids)} test (final numbers, never used for a choice)")
    print(f"Intervals: 95 % bootstrap over test messages, {N_BOOT} resamples, paired across policies\n")

    candidates = {f"{m} gate {tau:g}s": (kind, fset, gi)
                  for m, (kind, fset) in MODELS.items() for gi, tau in enumerate(GATES)}
    summary, curve_rows = [], []
    boot_rng = np.random.default_rng(SEED + 1)

    for grp in [MIX] + GRANS:
        idx_all, wt_all = group_index(S, grp)
        if len(idx_all) == 0:
            continue
        use_ks = S["use_ks"]["partial_word" if grp == MIX else grp]
        idx, wt = idx_all, wt_all
        dev_loc = np.where(dev_rows[idx])[0]
        test_loc = np.where(test_rows[idx])[0]
        if len(dev_loc) < 50 or len(test_loc) < 50:
            print(f"== {grp}: too few examples for a dev/test split, skipped ==\n")
            continue

        # 1) choose the learned variant on the dev half
        dev_scores = {n: dev_cv_score(S, idx, wt, use_ks, dev_loc, *c) for n, c in candidates.items()}
        chosen = min(dev_scores, key=dev_scores.get)

        # 2) final scoring on the test half
        idx_t, wt_t = idx[test_loc], wt[test_loc]
        fams, models = {}, {}
        for f, arr in rule_curves(S, idx_t, wt_t).items():
            fams[f] = arr
        for n, (kind, fset, gi) in candidates.items():
            fams[n], models[n], _ = run_variant(S, idx, wt, use_ks, dev_loc, test_loc, kind, fset, gi)
        abl = {}
        for n, (kind, fset) in ABLATIONS.items():
            abl[n + " gate 0s"], models[n], _ = run_variant(S, idx, wt, use_ks, dev_loc, test_loc, kind, fset, 0)
        always = stats_matrix(S, idx_t, wt_t, np.ones(len(idx_t)), S["base"][idx_t]).sum(0)
        all_f = dict(fams)
        all_f.update(abl)
        good_share = 100 * float((wt_t * S["good"][idx_t]).sum() / wt_t.sum())
        point = {f: a.sum(0) for f, a in all_f.items()}
        for f, a in point.items():
            for k in range(a.shape[0]):
                par = RULE_PARAMS[f][k] if f in RULE_PARAMS else float(QUANTILES[k])
                curve_rows.append(dict(group=grp, family=f, point=k, param=par,
                                       calls=100 * a[k, 1] / a[k, 0], missed=100 * a[k, 3] / a[k, 2]))

        # 3) paired bootstrap over test messages
        pass_fams = [chosen] + RULE_FAMILIES
        boots = {f: np.zeros((N_BOOT, len(BUDGETS))) for f in all_f}
        pboot = {f: np.zeros(N_BOOT) for f in pass_fams}
        for b in range(N_BOOT):
            ix = boot_rng.choice(test_ids, len(test_ids))
            alw = None
            for f, a in all_f.items():
                tot = a[ix].sum(0)
                for bi, bud in enumerate(BUDGETS):
                    boots[f][b, bi] = read_out(tot, bud)
                if f in pass_fams:
                    if alw is None:
                        alw = stats_matrix(S, idx_t, wt_t, np.ones(len(idx_t)), S["base"][idx_t])[ix].sum(0)
                    pboot[f][b] = pass_calls(tot, alw)

        rand = {b: 100.0 - b for b in BUDGETS}
        orc = {b: max(0.0, 100.0 * (1.0 - b / good_share)) for b in BUDGETS}
        print(f"== {grp}: always-invoke delivers at {good_share:.1f} % of test decision points "
              f"(the oracle needs that few calls) ==")
        print("Choice on the dev half (mean missed % over the budgets, lower is better):")
        for n, v in sorted(dev_scores.items(), key=lambda kv: kv[1]):
            print(f"   {n:<38}{v:>6.1f}{'   <- chosen' if n == chosen else ''}")
        print(f"\nTEST half, missed % at each call budget [95 % interval]:")
        print(f"{'policy':<44}" + "".join(f"{'@' + str(b) + '% calls':>20}" for b in BUDGETS))
        print(f"{'random skip':<44}" + "".join(f"{rand[b]:>20.1f}" for b in BUDGETS))
        print(f"{'oracle':<44}" + "".join(f"{orc[b]:>20.1f}" for b in BUDGETS))
        for f in all_f:
            tag = "* " if f == chosen else "  "
            cells = []
            for bi, bud in enumerate(BUDGETS):
                est = read_out(point[f], bud)
                lo, hi = pct(boots[f][:, bi])
                cells.append(fmt(est, lo, hi))
                summary.append(dict(group=grp, family=f, chosen=(f == chosen), budget=bud, missed=est,
                                    lo=lo, hi=hi))
            print(f"{tag + f:<44}" + "".join(f"{c:>20}" for c in cells))
        print("  (* = variant chosen on the dev half; the other learned rows are for reference only)")

        rule_names = [f for f in fams if f in RULE_FAMILIES]
        print(f"\nChosen variant ({chosen}) vs the best simple rule at each budget "
              f"(the rule may pick its best family, the learned policy may not):")
        for bi, bud in enumerate(BUDGETS):
            denom = rand[bud] - orc[bud]
            ml = read_out(point[chosen], bud)
            cand = [(read_out(point[f], bud), f) for f in rule_names]
            cand = [(m, f) for m, f in cand if np.isfinite(m)]
            if not np.isfinite(ml) or not cand or denom <= 0:
                print(f"  {bud}% calls: not available (a curve does not reach this call budget)")
                continue
            mr, best_r = min(cand)
            lo, hi = pct(boots[chosen][:, bi] - boots[best_r][:, bi])
            print(f"  {bud}% calls: gap closed {100 * (rand[bud] - ml) / denom:.0f} % vs "
                  f"{100 * (rand[bud] - mr) / denom:.0f} % ({best_r}); learned minus rule in missed points "
                  f"{ml - mr:+.1f} [{lo:+.1f}, {hi:+.1f}]")

        print(f"\nProposed selection rule (calls <= {MAX_CALLS_PCT:.0f} %, missed <= {MAX_MISSED_PCT:.0f} %, "
              f"value/call >= {MIN_VPC_RATIO} x always):")
        for f in pass_fams:
            x = pboot[f]
            share = 100 * float(np.isfinite(x).mean())
            if np.isfinite(x).any():
                lo, hi = pct(x)
                extra = f"fewest calls that pass: median {np.nanmedian(x):.0f} % [{lo:.0f}-{hi:.0f}]"
            else:
                extra = "never passes at <= 50 % calls"
            print(f"  {f:<38} passes in {share:.0f} % of resamples; {extra}")
            summary.append(dict(group=grp, family=f, chosen=(f == chosen), budget="pass",
                                missed=share, lo=np.nan, hi=np.nan))

        print("\nFeature check (test half, missed % @35% / @50% calls, no wait):")
        for f in ["logreg [word shape] gate 0s", "logreg [prefix length] gate 0s",
                  "logreg [word shape+length] gate 0s", "logreg [timing+position, no length] gate 0s",
                  "logreg [timing+position] gate 0s", "logreg [timing] gate 0s"]:
            if f not in point:
                continue
            c = [read_out(point[f], b) for b in (35, 50)]
            print(f"  {f:<44}" + "".join(f"{v:>9.1f}" if np.isfinite(v) else f"{'n/a':>9}" for v in c))
        m_full = models.get("logreg [timing+position] gate 0s")
        if m_full is not None:
            coef = m_full[-1].coef_[0]
            order = np.argsort(-np.abs(coef))[:5]
            print("  largest standardised coefficients of logreg [timing+position]: " +
                  ", ".join(f"{FEATURE_SETS['timing+position'][i]} {coef[i]:+.2f}" for i in order))
        print()
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    pd.DataFrame(curve_rows).to_csv(CURVES, index=False)
    print(f"Saved {SUMMARY} and {CURVES}")


if __name__ == "__main__":
    main()