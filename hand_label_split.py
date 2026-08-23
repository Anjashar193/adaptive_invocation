import csv
import random
import os
import sys

INPUT_CSV = "phase1_results_v2.csv"
SAMPLES_PER_GRANULARITY = 75   # 75 x 4 = 300 total, split 150/150 between two labelers
random.seed(42)                # same seed for both labelers -> same split every time


def build_full_sample():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_granularity = {}
    for row in rows:
        by_granularity.setdefault(row["granularity"], []).append(row)

    sample = []
    for granularity, group in by_granularity.items():
        group_sorted = sorted(group, key=lambda r: (r["prefix"], r["system"]))
        random.Random(42).shuffle(group_sorted)  # fixed seed -> identical order every run
        sample.append((granularity, group_sorted[:SAMPLES_PER_GRANULARITY]))
    return sample


def split_for_labeler(full_sample, labeler):
    my_rows = []
    for granularity, group in full_sample:
        half = len(group) // 2
        if labeler == "A":
            my_rows.extend(group[:half])
        else:
            my_rows.extend(group[half:])
    random.Random(1 if labeler == "A" else 2).shuffle(my_rows)
    return my_rows


def load_existing_labels(output_csv):
    if not os.path.exists(output_csv):
        return {}
    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["granularity"], r["system"], r["prefix"], r["prediction"]): r for r in rows}


def save_label(output_csv, row, manual_score, labeler):
    file_exists = os.path.exists(output_csv)
    fieldnames = ["granularity", "system", "prefix", "reference", "prediction",
                  "usefulness_auto", "manual_score", "labeler"]
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "granularity": row["granularity"],
            "system": row["system"],
            "prefix": row["prefix"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "usefulness_auto": row["usefulness"],
            "manual_score": manual_score,
            "labeler": labeler,
        })


def main():
    labeler = None
    if len(sys.argv) > 1 and sys.argv[1].upper() in ("A", "B"):
        labeler = sys.argv[1].upper()
    else:
        while labeler not in ("A", "B"):
            labeler = input("Which labeler are you? Type A or B: ").strip().upper()

    output_csv = f"hand_labels_{labeler}.csv"

    full_sample = build_full_sample()
    my_rows = split_for_labeler(full_sample, labeler)
    done = load_existing_labels(output_csv)

    remaining = [r for r in my_rows
                 if (r["granularity"], r["system"], r["prefix"], r["prediction"]) not in done]

    print(f"\nLabeler {labeler}: {len(done)} already labeled, {len(remaining)} left.\n")
    print("For each example: type 0 (irrelevant), 1 (close), or 2 (relevant).")
    print("Type 's' to skip, 'q' to save and quit.\n")

    for i, row in enumerate(remaining):
        print("-" * 70)
        print(f"[{i + 1}/{len(remaining)}]  granularity: {row['granularity']}  |  system: {row['system']}")
        print(f"PREFIX:     ...{row['prefix'][-100:]}")
        print(f"PREDICTION: {row['prediction']}")
        print(f"REFERENCE:  {row['reference']}")

        while True:
            answer = input("Your score (0/1/2, s=skip, q=quit): ").strip().lower()
            if answer == "q":
                print(f"\nSaved to {output_csv}. Run again with '{labeler}' to continue.")
                return
            if answer == "s":
                break
            if answer in ("0", "1", "2"):
                save_label(output_csv, row, answer, labeler)
                break
            print("Please type 0, 1, 2, s, or q.")

    print(f"\nAll of labeler {labeler}'s examples are done! Saved to {output_csv}.")


if __name__ == "__main__":
    main()