"""Phase 3a: copy of the Phase 1 results with the latency replaced by the llama.cpp measurement.

    uv run python -m src.swap_latency Q4_K_M_all

Reads  phase1_results_v3.csv  and  results/llama_bench_<label>.csv  (from src/bench_llama.py, cache run).
Writes phase1_results_v3_llama_<label>.csv: same rows, same suggestions and scores, but
latency_top1_ms of the Qwen rows is the llama.cpp time for that prefix.
The policy scripts can then be run on it:
    RESULTS=phase1_results_v3_llama_<label>.csv uv run python -m src.eval_frozen

Limit: the suggestions and their scores are still the Phase 1 (Hugging Face) ones; only the time changes.
Rows without a llama.cpp measurement are dropped, with a count printed.
"""
import csv
import os
import sys

SRC = os.environ.get("RESULTS", "phase1_results_v3.csv")
MODE = os.environ.get("MODE", "cache")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "Q4_K_M_all"
    bench = f"results/llama_bench_{label}.csv"
    lat = {}
    with open(bench, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["mode"] == MODE:
                lat[(r["message_id"], r["granularity"], r["cut_index"])] = r["latency_ms"]
    with open(SRC, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        rows = list(rd)
    out, missing, msgs = [], 0, set()
    covered = {k[0] for k in lat}
    for r in rows:
        if r["message_id"] not in covered:
            continue
        if r["system"] == "qwen":
            key = (r["message_id"], r["granularity"], r["cut_index"])
            if key not in lat:
                missing += 1
                continue
            r = dict(r, latency_top1_ms=lat[key])
        out.append(r)
        msgs.add(r["message_id"])
    stem = os.path.splitext(os.path.basename(SRC))[0]
    path = f"{stem}_llama_{label}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    nq = sum(r["system"] == "qwen" for r in out)
    print(f"{len(msgs)} messages, {nq} Qwen rows with llama.cpp latency ({MODE} run); {missing} rows without a measurement dropped")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
