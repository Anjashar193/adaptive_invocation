"""
Recompute metrics (especially `usefulness`) from an existing phase1_results.csv,
using the fixed trim-aware scoring logic in metrics.py - without re-running the
model. The raw predictions are already saved, so this just re-scores them.
"""

import csv
from metrics import score_example

INPUT_CSV = "phase1_results.csv"
OUTPUT_CSV = "phase1_results_v2.csv"


def recompute(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        granularity = row["granularity"]
        prediction = row["prediction"]
        example = {"reference": row["reference"], "full_text": row.get("prefix", "")}
        scores = score_example(prediction, example, granularity)
        row.update({k: str(v) for k, v in scores.items()})

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recomputed {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    recompute()