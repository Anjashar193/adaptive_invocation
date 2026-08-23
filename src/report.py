import csv
from collections import defaultdict

METRIC_COLUMNS = {
    "partial_word": ["char_match"],
    "next_word": ["first_word_match"],
    "phrase": ["semantic_sim", "overlap"],
    "sentence": ["semantic_sim", "overlap"],
}

TOPK_GRANULARITIES = {"next_word", "partial_word"}
MIN_REFERENCE_LEN_FOR_TOPK = 3  # below this, partial_word "hits" are often coincidental (e.g. ref="y" matches "you")


def load_results(path="phase1_results.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _avg(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["granularity"], row["system"])].append(row)

    header = f"{'granularity':<14}{'system':<20}{'n':<6}{'latency(ms)':<14}{'usefulness':<12}{'top5_acc':<10}{'top5_acc(len3+)':<16}metrics"
    print(header)
    print("-" * len(header))

    for (granularity, system), group in sorted(groups.items()):
        n = len(group)
        avg_latency = sum(float(r["latency_ms"]) for r in group) / n if n else 0
        avg_usefulness = _avg(group, "usefulness")

        top5_acc = ""
        top5_acc_filtered = ""
        if granularity in TOPK_GRANULARITIES:
            topk_rows = [r for r in group if r.get("topk_hit") not in (None, "")]
            if topk_rows:
                top5_acc = f"{_avg(topk_rows, 'topk_hit'):.3f}"

                if granularity == "partial_word":
                    filtered = [r for r in topk_rows if len(r["reference"]) >= MIN_REFERENCE_LEN_FOR_TOPK]
                    if filtered:
                        top5_acc_filtered = f"{_avg(filtered, 'topk_hit'):.3f} (n={len(filtered)})"

        metric_strs = []
        for col in METRIC_COLUMNS.get(granularity, []):
            metric_strs.append(f"{col}={_avg(group, col):.3f}")

        print(f"{granularity:<14}{system:<20}{n:<6}{avg_latency:<14.1f}{avg_usefulness:<12.3f}"
              f"{top5_acc:<10}{top5_acc_filtered:<16}{' '.join(metric_strs)}")


if __name__ == "__main__":
    rows = load_results()
    summarize(rows)