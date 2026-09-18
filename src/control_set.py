"""Build the stratified control set that decides the per-granularity metric.

Design (PHASE2_METRIC_PLAN.md Part 2):

* **Size is set by power, not convenience.** Subsampling the existing labels
  and re-running the decision shows partial_word detects its own true effect
  only ~58% of the time at n=60 - a coin flip. It needs ~120; the others are
  adequate at 75.
* **Sampling is stratified**, because uniform sampling under-represents the
  cases that actually decide the question. The gain is real but modest
  (~1.04-1.12x), so stratification supplements the size choice, it does not
  replace it.
* **Inclusion probabilities are stored** so every reported rate can be
  Horvitz-Thompson reweighted back to the uniform population. Stratified
  sampling biases headline numbers by construction; the correction lives in
  code, not in the discussion section.
"""

import csv
import random

try:
    from label_join import attach_similarity
    from metrics import keystrokes_saved
except ImportError:
    from src.label_join import attach_similarity
    from src.metrics import keystrokes_saved

OUTPUT_CSV = "control_set.csv"

# partial_word carries the weakest true effect and binds the whole design.
TARGET_SIZE = {"partial_word": 120, "next_word": 75, "phrase": 75, "sentence": 75}

STRATUM_SHARE = {"disagreement": 0.40, "paraphrase": 0.35, "anchor": 0.25}

SEED = 42


def dedupe_by_source_message(rows, min_shared_length=25, prefix_key="prefix"):
    """Keep one example per source message.

    Five cut points of the same message are not independent observations; if
    they land on both sides of a comparison the bootstrap intervals come out
    too narrow. Shorter prefixes are kept and anything extending them dropped.
    """
    kept, kept_prefixes, seen = [], [], set()
    for row in sorted(rows, key=lambda r: len(r[prefix_key])):
        prefix = row[prefix_key]
        if prefix in seen:  # exact repeats: same prefix, different sampled cut
            continue
        if any(len(k) >= min_shared_length and prefix.startswith(k) for k in kept_prefixes):
            continue
        kept.append(row)
        kept_prefixes.append(prefix)
        seen.add(prefix)
    return kept


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


def assign_strata(items):
    """Split one granularity's pool into the three sampling strata.

    disagreement - the two metrics rank the item very differently; this is
                   where the decision is actually made.
    paraphrase   - zero literal overlap but high similarity; decides whether
                   the symbolic metric is admissible on long granularities.
    anchor       - everything else, sampled uniformly to estimate the bias
                   introduced by the other two strata.
    """
    sim_ranks = _rank([i["sim"] for i in items])
    ks_ranks = _rank([i["keystrokes"] for i in items])
    for item, sim_rank, ks_rank in zip(items, sim_ranks, ks_ranks):
        item["rank_gap"] = abs(sim_rank - ks_rank)

    ordered = sorted(items, key=lambda i: -i["rank_gap"])
    cut = max(1, len(ordered) // 3)
    disagreement = set(id(i) for i in ordered[:cut])

    median_sim = sorted(i["sim"] for i in items)[len(items) // 2]
    for item in items:
        if id(item) in disagreement:
            item["stratum"] = "disagreement"
        elif item["keystrokes"] == 0 and item["sim"] >= median_sim:
            item["stratum"] = "paraphrase"
        else:
            item["stratum"] = "anchor"
    return items


def sample_stratified(items, target, seed=SEED):
    """Draw `target` items, recording each one's inclusion probability."""
    rng = random.Random(seed)
    by_stratum = {}
    for item in items:
        by_stratum.setdefault(item["stratum"], []).append(item)

    selected = []
    for stratum, share in STRATUM_SHARE.items():
        pool = by_stratum.get(stratum, [])
        if not pool:
            continue
        want = min(len(pool), round(target * share))
        drawn = rng.sample(pool, want)
        for item in drawn:
            # P(selected | in this stratum) - the Horvitz-Thompson weight is 1/p
            item["inclusion_prob"] = want / len(pool)
            item["ht_weight"] = len(pool) / want
            selected.append(item)

    # Top up from whatever remains if rounding left us short.
    if len(selected) < target:
        chosen = {id(i) for i in selected}
        rest = [i for i in items if id(i) not in chosen]
        for item in rng.sample(rest, min(len(rest), target - len(selected))):
            item["inclusion_prob"] = 1.0
            item["ht_weight"] = 1.0
            selected.append(item)

    rng.shuffle(selected)
    return selected


def _load_pool(results_csv, cache_path, system="qwen"):
    """All candidate items, with similarity attached.

    Drawn from the full results CSV rather than only the already-labelled
    rows: the control set is a fresh sample, and restricting it to the 299
    previously labelled items would both exhaust the pool and inherit that
    round's sampling.
    """
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["system"] == system]

    entries = [{"result": r} for r in rows]
    attach_similarity(entries, cache_path)

    pool = []
    for entry in entries:
        result = entry["result"]
        pool.append({
            "granularity": result["granularity"],
            "prefix": result["prefix"],
            "reference": result["reference"],
            "prediction": result["prediction"],
            "topk_predictions": result.get("topk_predictions", ""),
            "sim": entry["sim"],
            "keystrokes": keystrokes_saved(result["prediction"], result["reference"]),
        })
    return pool


def build_control_set(output_csv=OUTPUT_CSV, results_csv="phase1_results_v2.csv",
                      cache_path="similarity_cache.csv", seed=SEED):
    all_items = _load_pool(results_csv, cache_path)

    rows = []
    for granularity, target in TARGET_SIZE.items():
        pool = [i for i in all_items if i["granularity"] == granularity]

        pool = dedupe_by_source_message(pool)
        assign_strata(pool)
        selected = sample_stratified(pool, min(target, len(pool)), seed=seed)

        print(f"{granularity:<14} pool={len(pool):3d} target={target:3d} "
              f"selected={len(selected):3d}"
              + ("  << pool exhausted, needs fresh items" if len(selected) < target else ""))
        rows.extend(selected)

    for n, row in enumerate(rows, start=1):
        row["item_id"] = f"c{n:04d}"

    fieldnames = ["item_id", "granularity", "prefix", "reference", "prediction",
                  "topk_predictions", "sim", "keystrokes", "stratum",
                  "inclusion_prob", "ht_weight"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} items to {output_csv}")
    return rows


if __name__ == "__main__":
    rows = build_control_set()
    counts = {}
    for row in rows:
        counts[(row["granularity"], row["stratum"])] = \
            counts.get((row["granularity"], row["stratum"]), 0) + 1
    print("\nStratum breakdown:")
    for key in sorted(counts):
        print(f"  {key[0]:<14}{key[1]:<14}{counts[key]:>4}")
