"""Hand-labelling CLI for the control set.

Three independently elicited axes, because a single 0/1/2 score conflates two
different judgments (see PHASE2_METRIC_PLAN.md Part 1):

  A1 plausible      reference HIDDEN - "could a writer continue this way?"
  A2 matches        reference SHOWN  - "does it agree with what was typed?"
  A3 preference     two candidates   - "which would you rather have seen?"

A1 runs as its own pass with the reference withheld. If the reference is
visible it anchors the plausibility judgment and the two axes stop being
separable, which is exactly the failure mode in the first labelling round.

Unlike the previous version this stores the FULL prefix and a stable
result_row_id, so labels can be joined back to the results CSV exactly rather
than reconstructed from a truncated prefix and a candidate list.
"""

import argparse
import csv
import os

CONTROL_SET_CSV = "control_set.csv"
OUTPUT_CSV = "control_labels.csv"

FIELDNAMES = ["item_id", "result_row_id", "granularity", "prefix", "reference",
              "prediction", "topk_predictions", "axis", "score", "labeler"]

AXES = {
    "plausible": {
        "prompt": "Could a competent writer continue this way? (0=no, 1=yes)",
        "valid": {"0", "1"},
        "show_reference": False,
    },
    "matches": {
        "prompt": "Does it agree with what was actually typed? "
                  "(0=unrelated, 1=same intent, 2=agrees)",
        "valid": {"0", "1", "2"},
        "show_reference": True,
    },
}


def load_existing(path=OUTPUT_CSV):
    """Labels already recorded, keyed by (item_id, axis, labeler)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["item_id"], r["axis"], r["labeler"]): r for r in csv.DictReader(f)}


def save_label(row, axis, score, labeler, path=OUTPUT_CSV):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "item_id": row["item_id"],
            "result_row_id": row.get("result_row_id", ""),
            "granularity": row["granularity"],
            "prefix": row["prefix"],            # full, never truncated
            "reference": row["reference"],
            "prediction": row["prediction"],
            "topk_predictions": row.get("topk_predictions", ""),
            "axis": axis,
            "score": score,
            "labeler": labeler,
        })


def label_pass(items, axis, labeler, path=OUTPUT_CSV):
    """Run one labelling pass over `items` for a single axis."""
    spec = AXES[axis]
    done = load_existing(path)
    todo = [i for i in items if (i["item_id"], axis, labeler) not in done]

    print(f"\n=== Pass: {axis} ===")
    print(spec["prompt"])
    if not spec["show_reference"]:
        print("The reference is HIDDEN on purpose - judge the prefix only.")
    print(f"{len(items) - len(todo)} already done, {len(todo)} to go.")
    print("'s' to skip, 'q' to save and quit.\n")

    for n, row in enumerate(todo, start=1):
        print("-" * 70)
        print(f"[{n}/{len(todo)}]  {row['granularity']}")
        print(f"PREFIX:     ...{row['prefix'][-160:]}")
        print(f"CANDIDATE:  {row['prediction']}")
        if spec["show_reference"]:
            print(f"REFERENCE:  {row['reference']}")

        while True:
            answer = input(f"{axis} {sorted(spec['valid'])} / s / q: ").strip().lower()
            if answer == "q":
                print(f"\nSaved to {path}. Re-run to continue.")
                return False
            if answer == "s":
                break
            if answer in spec["valid"]:
                save_label(row, axis, answer, labeler, path)
                break
            print(f"  please type one of {sorted(spec['valid'])}, s, or q.")
    return True


def load_control_set(path=CONTROL_SET_CSV):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Build it first:\n"
            f"    uv run python -m src.control_set")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _axis_progress(items, axis, labeler, path=OUTPUT_CSV):
    done = load_existing(path)
    return sum(1 for i in items if (i["item_id"], axis, labeler) in done)


def main():
    parser = argparse.ArgumentParser(description="Label the control set, one axis at a time.")
    parser.add_argument("--axis", choices=sorted(AXES), help="which axis to label")
    parser.add_argument("--labeler", default="A", help="labeler id (use a second id for IAA)")
    parser.add_argument("--limit", type=int, help="stop after this many items")
    args = parser.parse_args()

    items = load_control_set()

    if not args.axis:
        print(__doc__)
        print(f"Control set: {len(items)} items.\n")
        print("Progress:")
        for axis in ("plausible", "matches"):
            for labeler in ("A", "B"):
                n = _axis_progress(items, axis, labeler)
                if n or labeler == "A":
                    print(f"  {axis:<12} labeler {labeler}: {n}/{len(items)}")
        print("\nRun one pass at a time, in this order:")
        print("  uv run python hand_label.py --axis plausible --labeler A")
        print("  uv run python hand_label.py --axis matches   --labeler A")
        print("\nA1 (plausible) MUST finish before A2 (matches) for a given")
        print("labeler, or the reference will have anchored the plausibility score.")
        return

    todo = items[: args.limit] if args.limit else items
    label_pass(todo, args.axis, args.labeler)


if __name__ == "__main__":
    main()
