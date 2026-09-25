"""Inter-annotator agreement between labeler A and labeler B, on the SAME
underlying items - not A vs. the automatic scorer (that's compare_label.py's
job), but A vs. a second human, which is what actually answers "is our
scorer bad, or is the task itself ambiguous?"

The 100 overlap items (c0346+) are duplicates of real items from the
original 345 - same (granularity, prefix, reference), different item_id.
This matches each labeler-B row back to its labeler-A twin and compares
their scores, separately per axis (plausible / matches) and per granularity.
"""

import csv
from collections import defaultdict

try:
    from agreement import kappa, agreement_rates
except ImportError:
    from src.agreement import kappa, agreement_rates

LABELS_CSV = "control_labels.csv"
GRANULARITIES = ["next_word", "partial_word", "phrase", "sentence"]
AXES = ["plausible", "matches"]


def load_and_match(labels_csv=LABELS_CSV):
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    a_rows = [r for r in rows if r["labeler"] == "A"]
    b_rows = [r for r in rows if r["labeler"] == "B"]

    # index A's rows by (granularity, prefix, reference, axis) for lookup
    a_index = {}
    for r in a_rows:
        key = (r["granularity"], r["prefix"], r["reference"], r["axis"])
        a_index.setdefault(key, []).append(r)

    matched = []
    unmatched_b = []
    ambiguous = 0
    for b in b_rows:
        key = (b["granularity"], b["prefix"], b["reference"], b["axis"])
        candidates = a_index.get(key)
        if not candidates:
            unmatched_b.append(b)
            continue
        if len(candidates) > 1:
            ambiguous += 1
        matched.append((candidates[0], b))

    return matched, unmatched_b, ambiguous


def summarize(matched):
    by_axis_gran = defaultdict(lambda: {"a": [], "b": []})
    for a_row, b_row in matched:
        key = (a_row["axis"], a_row["granularity"])
        by_axis_gran[key]["a"].append(int(a_row["score"]))
        by_axis_gran[key]["b"].append(int(b_row["score"]))

    print(f"{'axis':<12}{'granularity':<14}{'n':<5}{'exact':<8}{'kappa':<8}")
    print("-" * 47)
    results = {}
    for axis in AXES:
        for granularity in GRANULARITIES:
            data = by_axis_gran.get((axis, granularity))
            if not data or len(data["a"]) < 3:
                n = len(data["a"]) if data else 0
                print(f"{axis:<12}{granularity:<14}{n:<5}(too few to summarize)")
                continue
            rates = agreement_rates(data["a"], data["b"])
            k = kappa(data["a"], data["b"])
            results[(axis, granularity)] = {"n": rates["n"], "exact": rates["exact"], "kappa": k}
            print(f"{axis:<12}{granularity:<14}{rates['n']:<5}{rates['exact']:<8.1%}{k:<8.3f}")
    return results


def main():
    matched, unmatched_b, ambiguous = load_and_match()
    print(f"Matched {len(matched)} labeler-B rows to their labeler-A twin.")
    if unmatched_b:
        print(f"Warning: {len(unmatched_b)} labeler-B rows had no matching A row.")
    if ambiguous:
        print(f"Note: {ambiguous} keys matched more than one A row - used the first.")
    print()
    summarize(matched)


if __name__ == "__main__":
    main()