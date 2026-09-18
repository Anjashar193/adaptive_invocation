import csv
from collections import defaultdict

try:  # works both as `python src/report.py` and as `from src.report import ...`
    from metrics import keystrokes_saved, chars_saved_per_second
except ImportError:
    from src.metrics import keystrokes_saved, chars_saved_per_second

METRIC_COLUMNS = {
    "partial_word": ["char_match"],
    "next_word": ["first_word_match"],
    "phrase": ["semantic_sim", "overlap"],
    "sentence": ["semantic_sim", "overlap"],
}

TOPK_GRANULARITIES = {"next_word", "partial_word", "phrase", "sentence"}
MIN_REFERENCE_LEN_FOR_TOPK = 3  # partial_word only: short remainders like "y" cause coincidental hits


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

    header = (f"{'granularity':<14}{'system':<20}{'n':<6}{'latency(ms)':<14}{'usefulness':<12}"
              f"{'ks_rate':<10}{'cs_per_sec':<12}{'ks_zero':<9}{'top5_acc':<10}{'top5_acc(len3+)':<16}metrics")
    print(header)
    print("-" * len(header))

    for (granularity, system), group in sorted(groups.items()):
        n = len(group)
        avg_latency = sum(float(r["latency_ms"]) for r in group) / n if n else 0
        avg_usefulness = _avg(group, "usefulness")

        # Symbolic metrics, derived at report time from columns already in the
        # CSV - deliberately not stored by score_example(), so the Phase 1
        # results file stays byte-identical.
        saved = [keystrokes_saved(r["prediction"], r["reference"]) for r in group]
        ref_chars = sum(len(r["reference"]) for r in group)
        ks_rate = sum(saved) / ref_chars if ref_chars else 0.0
        ks_zero = sum(1 for s in saved if s == 0) / n if n else 0.0
        cs_per_sec = chars_saved_per_second(sum(saved) / n, avg_latency) if n else 0.0

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

        cs_str = f"{cs_per_sec:.2f}" if avg_latency > 0 else "-"
        print(f"{granularity:<14}{system:<20}{n:<6}{avg_latency:<14.1f}{avg_usefulness:<12.3f}"
              f"{ks_rate:<10.3f}{cs_str:<12}{ks_zero:<9.2f}"
              f"{top5_acc:<10}{top5_acc_filtered:<16}{' '.join(metric_strs)}")

    print()
    print("ks_rate    = chars saved / reference chars. NOT comparable across")
    print("             granularities - median reference length differs ~9x")
    print("             (4 chars next/partial word vs 25 phrase, 35 sentence).")
    print("cs_per_sec = chars saved per second of compute. Comparable across")
    print("             granularities; this is what a policy optimizes.")
    print("ks_zero    = fraction with zero literal overlap. The symbolic")
    print("             metric's blind spot - valid paraphrases land here.")


if __name__ == "__main__":
    rows = load_results()
    summarize(rows)