import csv
from collections import defaultdict


METRIC_COLUMNS = {
    "partial_word": ["char_match", "usefulness"],
    "next_word": ["first_word_match", "usefulness"],
    "phrase": ["semantic_sim", "overlap", "usefulness"],
    "sentence": ["semantic_sim", "overlap", "usefulness"],
}


def load_results(path="phase1_results.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["granularity"], row["system"])].append(row)

    print(f"{'granularity':<14}{'system':<20}{'n':<6}{'avg latency (ms)':<18}metrics")
    print("-" * 90)
    for (granularity, system), group in sorted(groups.items()):
        n = len(group)
        avg_latency = sum(float(r["latency_ms"]) for r in group) / n if n else 0
        metric_strs = []
        for col in METRIC_COLUMNS.get(granularity, []):
            vals = [float(r[col]) for r in group if r.get(col) not in (None, "")]
            avg = sum(vals) / len(vals) if vals else 0
            metric_strs.append(f"{col}={avg:.3f}")
        print(f"{granularity:<14}{system:<20}{n:<6}{avg_latency:<18.1f}{' '.join(metric_strs)}")


if __name__ == "__main__":
    rows = load_results()
    summarize(rows)