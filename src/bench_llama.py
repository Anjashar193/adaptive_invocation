"""Phase 3a: latency of the same Qwen model under llama.cpp (GGUF), on the Phase 1 prefixes.

Start the server first, in another terminal (one quantization at a time):
    llama-server -m <path>/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 -c 2048

Then, from the repo root:
    uv run python -m src.bench_llama Q4_K_M
    uv run python -m src.bench_llama Q8_0        # after restarting the server with the Q8_0 file

What it does
  * takes the Qwen rows of phase1_results_v3.csv for the first MESSAGES messages (default 20)
  * sends each prefix to the server as raw text (no chat template), with the Phase 1 sampling
    settings (temperature 0.8, top_k 50, top_p 0.95; min_p off) and the Phase 1 stopping rules
    (a completed word, 5 words, or a full sentence), applied on the streamed text
  * measures the time until the suggestion is complete (one suggestion, as a user would wait)
  * does this twice: without the prompt cache, then with it (prefixes in typing order)
  * prints medians per granularity next to the Phase 1 (Hugging Face) time for the same rows
Writes results/llama_bench_<label>.csv (one row per request) and results/llama_bench_<label>_summary.csv

Environment: MESSAGES (default 20), URL (default http://127.0.0.1:8080), RESULTS (default phase1_results_v3.csv)
"""
import csv
import http.client
import json
import os
import re
import statistics
import sys
import time
from urllib.parse import urlparse

MAX_TOKEN_CEILING = {"partial_word": 10, "next_word": 10, "phrase": 25, "sentence": 60}   # as in run_eval.py
GRANULARITIES = ["partial_word", "next_word", "phrase", "sentence"]
URL = os.environ.get("URL", "http://127.0.0.1:8080")
RESULTS = os.environ.get("RESULTS", "phase1_results_v3.csv")
MESSAGES = int(os.environ.get("MESSAGES", "20"))
SEED = 42


def should_stop(text, granularity):
    """Same rule as GranularityStop in run_eval.py."""
    stripped = text.strip()
    if not stripped:
        return False
    if granularity in ("partial_word", "next_word"):
        if len(stripped.split()) >= 2:
            return True
        return bool(re.search(r"[,.!?;:]", stripped))
    if granularity == "phrase":
        return len(stripped.split()) >= 5
    if granularity == "sentence":
        return bool(re.search(r"[.!?](\s|$)", text))
    return False


def finish(text, granularity):
    text = text.strip()
    if granularity in ("partial_word", "next_word") and text:
        text = text.split()[0]
    return text


def complete(prefix, granularity, seed, cache):
    """One request. Returns (suggestion, ms until complete, ms until first token, tokens, prompt tokens from cache)."""
    u = urlparse(URL)
    body = json.dumps({
        "prompt": prefix, "n_predict": MAX_TOKEN_CEILING[granularity], "stream": True,
        "temperature": 0.8, "top_k": 50, "top_p": 0.95, "min_p": 0.0,
        "seed": seed, "cache_prompt": cache,
    })
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
    t0 = time.perf_counter()
    conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"server answered {resp.status}: {resp.read()[:200]!r}")
    text, n_tok, t_first, cache_n = "", 0, None, None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        d = json.loads(line[5:])
        piece = d.get("content", "")
        if piece:
            if t_first is None:
                t_first = time.perf_counter() - t0
            text += piece
            n_tok += 1
        if d.get("stop"):
            cache_n = (d.get("timings") or {}).get("cache_n")
            break
        if should_stop(text, granularity):
            break                       # closing the connection cancels the generation on the server
    total = time.perf_counter() - t0
    conn.close()
    return finish(text, granularity), total * 1000, (t_first if t_first is not None else total) * 1000, n_tok, cache_n


def first_word(s):
    w = s.strip().split()
    return w[0].lower().strip(".,!?;:") if w else ""


def load_rows():
    with open(RESULTS, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["system"] == "qwen"]
    ids = sorted({int(r["message_id"]) for r in rows})[:MESSAGES]
    keep = set(ids)
    rows = [r for r in rows if int(r["message_id"]) in keep]
    order = {g: i for i, g in enumerate(GRANULARITIES)}
    rows.sort(key=lambda r: (int(r["message_id"]), order[r["granularity"]], int(r["cut_index"])))   # typing order
    return rows, len(ids)


def med(x):
    return statistics.median(x) if x else float("nan")


def p90(x):
    if not x:
        return float("nan")
    x = sorted(x)
    return x[min(len(x) - 1, int(round(0.9 * (len(x) - 1))))]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    rows, n_msg = load_rows()
    print(f"Source: {RESULTS}; first {n_msg} messages, {len(rows)} Qwen examples; server {URL}; label {label}")
    try:
        complete("Hello, this is a", "next_word", 1, False)      # warm-up, also checks the server is up
    except OSError as e:
        sys.exit(f"Cannot reach the server at {URL} ({e}). Start llama-server first (see the top of this file).")

    out = []
    for mode, cache in (("no_cache", False), ("cache", True)):
        t0 = time.time()
        for i, r in enumerate(rows):
            seed = SEED * 1000003 + int(r["message_id"]) * 100 + GRANULARITIES.index(r["granularity"]) * 10 + int(r["cut_index"])
            sug, ms, ms_first, n_tok, cache_n = complete(r["prefix"], r["granularity"], seed, cache)
            out.append({
                "label": label, "mode": mode, "message_id": r["message_id"], "cut_index": r["cut_index"],
                "granularity": r["granularity"], "prefix_words": r["prefix_words"],
                "latency_ms": round(ms, 2), "first_token_ms": round(ms_first, 2), "tokens": n_tok,
                "cache_n": "" if cache_n is None else cache_n,
                "hf_latency_top1_ms": r["latency_top1_ms"],
                "suggestion": sug, "hf_prediction": r["prediction"], "reference": r["reference"],
            })
        print(f"  {mode}: {len(rows)} requests in {time.time() - t0:.0f} s")

    os.makedirs("results", exist_ok=True)
    path = f"results/llama_bench_{label}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print()
    print("Time until one suggestion is complete, ms (median; 90th percentile in brackets)")
    print(f"{'granularity':<14}{'n':>5}{'HF fp32 (Phase 1)':>20}{'llama.cpp no cache':>22}{'llama.cpp cache':>20}{'tokens':>8}{'1st word = ref, HF / llama':>30}")
    summary = []
    for g in GRANULARITIES:
        sel = [o for o in out if o["granularity"] == g]
        nc = [o for o in sel if o["mode"] == "no_cache"]
        c = [o for o in sel if o["mode"] == "cache"]
        hf = [float(o["hf_latency_top1_ms"]) for o in nc]
        l_nc, l_c = [o["latency_ms"] for o in nc], [o["latency_ms"] for o in c]
        if g == "partial_word":       # the reference is the rest of the word
            hit_hf = [o["hf_prediction"].strip().lower() == o["reference"].strip().lower() for o in nc]
            hit_ll = [o["suggestion"].strip().lower() == o["reference"].strip().lower() for o in sel]
        else:
            hit_hf = [first_word(o["hf_prediction"]) == first_word(o["reference"]) for o in nc]
            hit_ll = [first_word(o["suggestion"]) == first_word(o["reference"]) for o in sel]
        row = {
            "label": label, "granularity": g, "n": len(nc),
            "hf_median_ms": round(med(hf), 1), "hf_p90_ms": round(p90(hf), 1),
            "no_cache_median_ms": round(med(l_nc), 1), "no_cache_p90_ms": round(p90(l_nc), 1),
            "cache_median_ms": round(med(l_c), 1), "cache_p90_ms": round(p90(l_c), 1),
            "median_tokens": med([o["tokens"] for o in sel]),
            "first_word_match_hf_pct": round(100 * sum(hit_hf) / max(1, len(hit_hf)), 1),
            "first_word_match_llama_pct": round(100 * sum(hit_ll) / max(1, len(hit_ll)), 1),
        }
        summary.append(row)
        print(f"{g:<14}{row['n']:>5}"
              f"{row['hf_median_ms']:>12.0f} [{row['hf_p90_ms']:.0f}]".ljust(39)
              + f"{row['no_cache_median_ms']:>10.0f} [{row['no_cache_p90_ms']:.0f}]".ljust(22)
              + f"{row['cache_median_ms']:>10.0f} [{row['cache_p90_ms']:.0f}]".ljust(20)
              + f"{row['median_tokens']:>8.0f}"
              + f"{row['first_word_match_hf_pct']:>16.1f} / {row['first_word_match_llama_pct']:.1f}")
    spath = f"results/llama_bench_{label}_summary.csv"
    with open(spath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print()
    print("Notes: one sampled suggestion per request, so the match column is noisy; it is a sanity check, not the Phase 1 metric.")
    print("       The cache run sends the cut points of a message in order; real typing extends the text one key at a time.")
    print(f"Saved {path} and {spath}")


if __name__ == "__main__":
    main()
