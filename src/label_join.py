"""Join hand labels back to the evaluation results.

`hand_labels.csv` stores a truncated prefix and a numbered candidate list, so
it cannot be joined on prefix. The reliable key is
(granularity, reference, normalized candidate tuple).

Splitting the candidate list on "|" is NOT safe: phrase and sentence
candidates contain literal "|" and "\\r\\n" characters, which drops the match
rate to 259/300. Splitting on the *numbering boundary* recovers 299/300.
"""

import csv
import os
import re

try:  # works both as `python src/label_join.py` and `from src.label_join import ...`
    from metrics import semantic_similarity, trim_prediction
except ImportError:
    from src.metrics import semantic_similarity, trim_prediction

# "1. foo | 2. bar" -> split before each "<digits>. ", not on every "|"
_CANDIDATE_BOUNDARY = re.compile(r"(?:^|\s\|\s)\d+\.\s")

EXPECTED_MATCHES = 299
EXPECTED_TOTAL = 300
# id=6 (partial_word, reference "20") has no counterpart in the results CSV.
KNOWN_UNMATCHED_IDS = {"6"}


def _squash(text):
    return " ".join(text.split())


def normalize_candidates_from_label(candidates):
    """Parse the labeller-facing "1. foo | 2. bar" column into a tuple."""
    parts = _CANDIDATE_BOUNDARY.split(candidates)
    return tuple(_squash(p) for p in parts if p.strip())


def normalize_candidates_from_results(topk_predictions):
    """Parse the results CSV `topk_predictions` column into a tuple."""
    return tuple(_squash(p) for p in topk_predictions.split(" | ") if p.strip())


def _key_from_label(row):
    return (row["granularity"], row["reference"],
            normalize_candidates_from_label(row["candidates"]))


def _key_from_result(row):
    return (row["granularity"], row["reference"],
            normalize_candidates_from_results(row["topk_predictions"]))


def join_labels_to_results(label_rows, result_rows, system="qwen"):
    """Attach each hand label to its originating result row.

    Returns (joined, unmatched). Each joined entry carries `match_count`:
    8 keys legitimately map to several result rows (same reference and same
    top-k list from different prefixes). All rows within such a group share
    the same `usefulness`, so collapsing to the first is safe - but the count
    is surfaced rather than hidden, so a future schema change that makes the
    key genuinely ambiguous is visible instead of silent.
    """
    index = {}
    for row in result_rows:
        if row["system"] != system:
            continue
        index.setdefault(_key_from_result(row), []).append(row)

    joined, unmatched = [], []
    for label in label_rows:
        group = index.get(_key_from_label(label))
        if not group:
            unmatched.append(label)
            continue
        joined.append({
            "label": label,
            "result": group[0],
            "match_count": len(group),
            "usefulness_conflict": len({r["usefulness"] for r in group}) > 1,
        })
    return joined, unmatched


def assert_join_health(joined, unmatched):
    """Fail loudly if the join degrades - these numbers back the thesis."""
    total = len(joined) + len(unmatched)
    if len(joined) < EXPECTED_MATCHES:
        ids = sorted(r.get("id", "?") for r in unmatched)
        raise AssertionError(
            f"join recovered {len(joined)}/{total}, expected >= {EXPECTED_MATCHES}. "
            f"Unmatched ids: {ids}. The candidate-splitting rule probably drifted."
        )
    conflicts = [j for j in joined if j["usefulness_conflict"]]
    if conflicts:
        raise AssertionError(
            f"{len(conflicts)} ambiguous keys map to rows with DIFFERENT usefulness; "
            "collapsing to the first row is no longer safe."
        )


def attach_similarity(joined, cache_path="similarity_cache.csv"):
    """Add `sim` to each joined row, caching embeddings across runs.

    `semantic_sim` is only populated for phrase/sentence in the results CSV -
    partial_word and next_word rows store char_match/first_word_match instead.
    Recomputing all 300 embeddings on every analysis run is the single biggest
    cost here, so results are cached by (granularity, prediction, reference).
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cache[(row["granularity"], row["prediction"], row["reference"])] = float(row["sim"])

    fresh = 0
    for entry in joined:
        result = entry["result"]
        key = (result["granularity"], result["prediction"], result["reference"])
        if key not in cache:
            trimmed = trim_prediction(result["prediction"], result["reference"],
                                      result["granularity"])
            cache[key] = semantic_similarity(trimmed, result["reference"])
            fresh += 1
        entry["sim"] = cache[key]

    if fresh:
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["granularity", "prediction", "reference", "sim"])
            writer.writeheader()
            for (granularity, prediction, reference), sim in sorted(cache.items()):
                writer.writerow({"granularity": granularity, "prediction": prediction,
                                 "reference": reference, "sim": sim})
    return joined


def load_joined(labels_csv="hand_labels.csv", results_csv="phase1_results_v2.csv",
                cache_path="similarity_cache.csv", with_similarity=True):
    """Load, join, health-check and (optionally) attach similarity."""
    with open(labels_csv, newline="", encoding="utf-8") as f:
        label_rows = list(csv.DictReader(f))
    with open(results_csv, newline="", encoding="utf-8") as f:
        result_rows = list(csv.DictReader(f))

    joined, unmatched = join_labels_to_results(label_rows, result_rows)
    assert_join_health(joined, unmatched)
    if with_similarity:
        attach_similarity(joined, cache_path)
    return joined, unmatched


if __name__ == "__main__":
    joined, unmatched = load_joined(with_similarity=False)
    print(f"joined {len(joined)}/{len(joined) + len(unmatched)}")
    print(f"unmatched ids: {sorted(r['id'] for r in unmatched)}")
    ambiguous = [j for j in joined if j["match_count"] > 1]
    print(f"ambiguous keys (>1 result row): {len(ambiguous)}")
    print(f"  ...with conflicting usefulness: {sum(j['usefulness_conflict'] for j in joined)}")
    print("candidate parsing, literal '|' inside a candidate:")
    print("  ", normalize_candidates_from_label("1. a | b | 2. c"))
