import csv
import random
import os

INPUT_CSV = "phase1_results_v2.csv"
OUTPUT_CSV = "hand_labels.csv"
SAMPLES_PER_GRANULARITY = 75  # 75 x 4 = 300 total


def dedupe_by_source_message(rows, min_shared_length=25):
    rows_sorted = sorted(rows, key=lambda r: len(r["prefix"]))
    kept = []
    kept_prefixes = []
    for row in rows_sorted:
        p = row["prefix"]
        is_continuation = any(
            len(kp) >= min_shared_length and p.startswith(kp)
            for kp in kept_prefixes
        )
        if not is_continuation:
            kept.append(row)
            kept_prefixes.append(p)
    return kept


def build_sample():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_granularity = {}
    for row in rows:
        if row["system"] != "qwen":
            continue  # calibration focuses on the model's own predictions
        by_granularity.setdefault(row["granularity"], []).append(row)

    sample = []
    for granularity, group in by_granularity.items():
        deduped = dedupe_by_source_message(group)
        group_sorted = sorted(deduped, key=lambda r: (r["prefix"], r["reference"]))
        random.Random(42).shuffle(group_sorted)
        sample.extend(group_sorted[:SAMPLES_PER_GRANULARITY])

    random.Random(7).shuffle(sample)
    return sample


def load_existing_labels():
    if not os.path.exists(OUTPUT_CSV):
        return {}
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["granularity"], r["prefix"], r["reference"]): r for r in rows}


def save_label(row, manual_score):
    file_exists = os.path.exists(OUTPUT_CSV)
    fieldnames = ["granularity", "prefix", "reference", "prediction",
                  "topk_predictions", "usefulness_auto", "topk_hit_auto",
                  "manual_score", "labeler"]
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "granularity": row["granularity"],
            "prefix": row["prefix"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "topk_predictions": row.get("topk_predictions", ""),
            "usefulness_auto": row["usefulness"],
            "topk_hit_auto": row.get("topk_hit", ""),
            "manual_score": manual_score,
            "labeler": "solo",
        })


def main():
    sample = build_sample()
    done = load_existing_labels()

    remaining = [r for r in sample
                 if (r["granularity"], r["prefix"], r["reference"]) not in done]

    print(f"{len(done)} already labeled, {len(remaining)} left to go.\n")
    print("Score 0 (irrelevant), 1 (close/plausible), or 2 (relevant/matches).")
    print("If a top-5 list is shown, score the BEST candidate in it.")
    print("Type 's' to skip, 'q' to save and quit.\n")

    for i, row in enumerate(remaining):
        print("-" * 70)
        print(f"[{i + 1}/{len(remaining)}]  granularity: {row['granularity']}")
        print(f"PREFIX:     ...{row['prefix'][-100:]}")

        topk = row.get("topk_predictions", "")
        if topk:
            candidates = [c.strip() for c in topk.split("|")]
            print("TOP-5 CANDIDATES:")
            for rank, c in enumerate(candidates, start=1):
                print(f"    {rank}. {c}")
        else:
            print(f"PREDICTION: {row['prediction']}")

        print(f"REFERENCE:  {row['reference']}")

        while True:
            answer = input("Your score (0/1/2, s=skip, q=quit): ").strip().lower()
            if answer == "q":
                print(f"\nSaved to {OUTPUT_CSV}. Run this script again anytime to continue.")
                return
            if answer == "s":
                break
            if answer in ("0", "1", "2"):
                save_label(row, answer)
                break
            print("Please type 0, 1, 2, s, or q.")

    print(f"\nAll {len(sample)} examples labeled! Run compare_labels.py next.")


if __name__ == "__main__":
    main()