"""Stage 6 - decide which metric Phase 2 optimizes: embeddings or keystrokes.

The decision rule is pre-registered in PHASE2_METRIC_PLAN.md Part 2.6 and
HOW_TO_ANSWER_THE_RESEARCH_QUESTION.md Step 6 - fixed before results are seen,
so it cannot be adjusted after the fact to favor a preferred answer.

Ground truth: the "matches" axis (reference shown, 0/1/2 agreement score).
`--axis preference` does not exist in hand_label.py - confirmed directly
against the CLI - so "matches" is the human judgment this decision is made
against, not a separate preference pass. (Flagged separately to the
supervisor - PHASE2_METRIC_PLAN.md's Stage 4 note says "A1/A2 passes" while
its own higher-level description says "A1/A2/A3"; the two disagree.)

IMPORTANT: control_labels.csv is read directly, NOT via label_join.py.
label_join.py's join logic expects a "candidates" column and re-derives
prediction/reference by fuzzy-matching against phase1_results_v2.csv - that
was built for the OLD hand_labels.csv format. control_set.csv (and therefore
control_labels.csv, which hand_label.py writes by copying those columns
straight through) already carries `reference`, `prediction` and
`topk_predictions` directly - there is nothing to rejoin. Attempting the old
join here would fail with a KeyError on a column that doesn't exist in this
file's real schema.

Rule, per granularity:
    paired bootstrap CI excludes 0  -> adopt the winner
    paired bootstrap CI includes 0  -> adopt keystrokes (simpler, no labels,
                                        no thresholds - a tie is broken by
                                        simplicity, never by the point estimate)
    bootstrap undetermined (not enough valid resamples, e.g. no variance in
    human scores) -> also adopt keystrokes, same reasoning as a tie. This
    case isn't explicitly named in the pre-registered rule, so it's flagged
    in the output rather than silently folded in.

Do NOT tune thresholds. Do NOT average the two metrics. Both were tested in
Part 1 and both made things worse (naive combination loses to the better
single metric everywhere; honest CV tuning makes sentence worse).
"""

import csv
import hashlib
import json
from datetime import datetime, timezone

try:
    from agreement import paired_bootstrap_spearman_delta, verdict, kappa, agreement_rates
    from metrics import keystrokes_saved, semantic_similarity, trim_prediction
except ImportError:
    from src.agreement import paired_bootstrap_spearman_delta, verdict, kappa, agreement_rates
    from src.metrics import keystrokes_saved, semantic_similarity, trim_prediction

OUTPUT_JSON = "metric_choice.json"
GRANULARITIES = ["next_word", "partial_word", "phrase", "sentence"]
MIN_LABELS_TO_DECIDE = 10
MIN_OVERLAP_FOR_KAPPA = 5  # below this, kappa is too noisy to report meaningfully


def compute_human_agreement(labels_csv, granularity, axis="matches"):
    """Inter-annotator kappa for one granularity, on the 'matches' axis (the
    same axis this whole decision is made against) - using whichever items
    labeler B double-labeled. Returns None if there's no overlap data for
    this granularity; not every granularity necessarily has one yet.

    This exists so a metric decision - especially a tie - carries its own
    evidence for whether that tie reflects real metric equivalence or just
    human disagreement about the underlying question. See
    PHASE2_METRIC_PLAN.md Part 1.3: "we cannot currently tell whether the
    scorer is bad or the task is ambiguous."
    """
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    a_rows = [r for r in rows if r["labeler"] == "A"]
    b_rows = [r for r in rows if r["labeler"] == "B"
              and r["granularity"] == granularity and r["axis"] == axis]

    a_index = {}
    for r in a_rows:
        key = (r["granularity"], r["prefix"], r["reference"], r["axis"])
        a_index.setdefault(key, []).append(r)

    a_scores, b_scores = [], []
    for b in b_rows:
        key = (b["granularity"], b["prefix"], b["reference"], b["axis"])
        candidates = a_index.get(key)
        if candidates:
            a_scores.append(int(candidates[0]["score"]))
            b_scores.append(int(b["score"]))

    if len(a_scores) < MIN_OVERLAP_FOR_KAPPA:
        return None

    rates = agreement_rates(a_scores, b_scores)
    return {"n": rates["n"], "exact_agreement": rates["exact"], "kappa": kappa(a_scores, b_scores)}


def load_matches_rows(labels_csv="control_labels.csv"):
    """Load the 'matches' axis rows directly - no join needed.

    control_labels.csv already carries granularity/reference/prediction for
    every row (copied straight from control_set.csv, which built them from
    phase1_results_v2.csv originally). The only filtering needed is by axis.
    """
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["axis"] == "matches"]


def compute_metrics_for_row(row, cache):
    """keystrokes_saved is cheap and computed fresh every time.
    semantic_similarity needs the embedding model, so it's cached by the
    same key used elsewhere in this project: (granularity, prediction, reference).
    """
    granularity = row["granularity"]
    prediction = row["prediction"]
    reference = row["reference"]

    ks = keystrokes_saved(prediction, reference)

    key = (granularity, prediction, reference)
    if key not in cache:
        trimmed = trim_prediction(prediction, reference, granularity)
        cache[key] = semantic_similarity(trimmed, reference)
    sim = cache[key]

    return ks, sim


def _fingerprint_source(labels_csv):
    """A short, stable fingerprint of the exact labeling data this decision
    was computed from - so if control_labels.csv is ever updated later,
    there's a way to tell whether a given metric_choice.json is still
    current, per PHASE2_METRIC_PLAN.md Stage 6: 'Emit versioned
    metric_choice.json'.
    """
    with open(labels_csv, "rb") as f:
        content = f.read()
    with open(labels_csv, newline="", encoding="utf-8") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    return {
        "source_file": labels_csv,
        "source_md5": hashlib.md5(content).hexdigest(),
        "source_row_count": n_rows,
    }


def decide_metric(labels_csv="control_labels.csv"):
    matches_rows = load_matches_rows(labels_csv)
    print(f"Loaded {len(matches_rows)} 'matches'-axis rows from {labels_csv}.")

    by_granularity = {g: [] for g in GRANULARITIES}
    for row in matches_rows:
        g = row["granularity"]
        if g in by_granularity:
            by_granularity[g].append(row)

    sim_cache = {}
    decisions = {}
    for granularity in GRANULARITIES:
        rows = by_granularity[granularity]
        if len(rows) < MIN_LABELS_TO_DECIDE:
            print(f"{granularity}: only {len(rows)} matches-axis labels - "
                  f"skipping, too few to decide (need >= {MIN_LABELS_TO_DECIDE}).")
            continue

        human, keystrokes, sims = [], [], []
        for row in rows:
            ks, sim = compute_metrics_for_row(row, sim_cache)
            human.append(int(row["score"]))
            keystrokes.append(ks)
            sims.append(sim)

        ci = paired_bootstrap_spearman_delta(human, keystrokes, sims)
        result = verdict(ci)

        if result == "a_better":
            chosen = "keystrokes"
        elif result == "b_better":
            chosen = "embeddings"
        else:
            # covers both "tie" and "undetermined" - pre-registered rule only
            # names "tie" explicitly; "undetermined" gets the same simple
            # default rather than silently defaulting without saying so.
            chosen = "keystrokes"
            if result == "undetermined":
                print(f"  note: {granularity} bootstrap was undetermined "
                      f"(not enough resamples had score variance) - "
                      f"defaulting to keystrokes, same as a tie.")

        decisions[granularity] = {
            "metric": chosen,
            "n": len(human),
            "delta": ci["delta"],
            "ci_lo": ci["lo"],
            "ci_hi": ci["hi"],
            "verdict": result,
        }

        human_agreement = compute_human_agreement(labels_csv, granularity, axis="matches")
        human_agreement_plausible = compute_human_agreement(labels_csv, granularity, axis="plausible")

        if human_agreement is not None:
            decisions[granularity]["human_agreement_matches"] = human_agreement
            if human_agreement["kappa"] < 0.2:
                decisions[granularity]["note"] = (
                    "Low inter-annotator agreement (kappa < 0.2) on the 'matches' axis "
                    "for this granularity - the axis this decision is actually made "
                    "against - so this verdict may reflect task ambiguity rather than "
                    "a real difference between the two candidate metrics."
                )

        if human_agreement_plausible is not None:
            decisions[granularity]["human_agreement_plausible"] = human_agreement_plausible
            if human_agreement_plausible["kappa"] < 0.2 and (
                    human_agreement is None or human_agreement["kappa"] >= 0.2):
                decisions[granularity]["note_plausible_axis"] = (
                    "Note: inter-annotator agreement is weak on the separate "
                    "'plausible' axis (kappa < 0.2) for this granularity, even though "
                    "agreement on 'matches' - the axis this decision actually depends "
                    "on - is solid. Worth reporting alongside this decision as a known "
                    "limitation of the plausibility judgment specifically."
                )

        lo_str = f"{ci['lo']:+.3f}" if ci["lo"] == ci["lo"] else "nan"
        hi_str = f"{ci['hi']:+.3f}" if ci["hi"] == ci["hi"] else "nan"
        agreement_str = ""
        if human_agreement is not None:
            agreement_str += f"  matches-kappa={human_agreement['kappa']:+.3f} (n={human_agreement['n']})"
        if human_agreement_plausible is not None:
            agreement_str += f"  plausible-kappa={human_agreement_plausible['kappa']:+.3f} (n={human_agreement_plausible['n']})"
        print(f"{granularity:<14} n={len(human):<4} delta={ci['delta']:+.3f} "
              f"[{lo_str}, {hi_str}]  verdict={result:<12} -> adopt {chosen}{agreement_str}")

    decisions["_metadata"] = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **_fingerprint_source(labels_csv),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)
    print(f"\nWrote decision to {OUTPUT_JSON}")
    return decisions


if __name__ == "__main__":
    decide_metric()