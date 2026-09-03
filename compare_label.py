import csv
from collections import Counter

LABELS_CSV = "hand_labels.csv"


def main():
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No labels found yet. Run hand_label_split.py first, then merge_labels.py.")
        return

    exact_matches = 0
    off_by_one = 0
    off_by_two = 0
    confusion = Counter()
    by_granularity = {}

    for row in rows:
        auto = int(float(row["usefulness_auto"]))
        manual = int(row["manual_score"])
        diff = abs(auto - manual)
        confusion[(manual, auto)] += 1

        g = row["granularity"]
        by_granularity.setdefault(g, {"match": 0, "total": 0})
        by_granularity[g]["total"] += 1
        if diff == 0:
            exact_matches += 1
            by_granularity[g]["match"] += 1
        elif diff == 1:
            off_by_one += 1
        else:
            off_by_two += 1

    n = len(rows)
    print(f"Total labeled: {n}\n")
    print(f"Exact match:        {exact_matches}/{n}  ({100 * exact_matches / n:.1f}%)")
    print(f"Off by 1:           {off_by_one}/{n}  ({100 * off_by_one / n:.1f}%)")
    print(f"Off by 2 (0 vs 2):  {off_by_two}/{n}  ({100 * off_by_two / n:.1f}%)\n")

    print("Agreement by granularity:")
    for g, d in by_granularity.items():
        print(f"  {g}: {d['match']}/{d['total']} ({100*d['match']/d['total']:.1f}%)")

    print("\nConfusion matrix (rows = your manual score, cols = code's auto score):")
    print(f"{'':>12}{'auto=0':>10}{'auto=1':>10}{'auto=2':>10}")
    for manual_val in [0, 1, 2]:
        row_str = f"manual={manual_val:<3}"
        for auto_val in [0, 1, 2]:
            row_str += f"{confusion.get((manual_val, auto_val), 0):>10}"
        print(row_str)

    print("\nInterpretation:")
    if exact_matches / n > 0.7:
        print("- Strong agreement. The 0.75/0.4 thresholds look well calibrated.")
    elif exact_matches / n > 0.5:
        print("- Moderate agreement. Consider a small threshold adjustment.")
    else:
        print("- Weak agreement. Thresholds likely need real adjustment.")


if __name__ == "__main__":
    main()