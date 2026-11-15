# Phase 3 — Runtime Validation

**One-line summary:** Under llama.cpp (Q4_K_M) one suggestion takes about a quarter to a half of the Phase 1 time on the same machine. Q8_0 was not faster. With the llama.cpp times, the frozen invocation rule gives nearly the same call and missed shares as before.

## Progress

- [x] **Phase 3a** — llama.cpp / GGUF: latency for Q4_K_M and Q8_0, with and without the prompt cache; frozen rule rerun with the llama.cpp times
- [ ] **Phase 3b** — LiteRT-LM (optional)

The invocation policy itself is described in `PHASE2A_POLICY_PLAN.md`.

---

## Phase 3a — llama.cpp

**Question.** How long does a suggestion take under a production-style runtime, and do the policy results change with those times?

**Setup.** The same model (Qwen3-0.6B) as a GGUF file, served by `llama-server` on the same machine. `src/bench_llama.py` sends the Phase 1 prefixes as raw text with the Phase 1 sampling settings (temperature 0.8, top_k 50, top_p 0.95) and the Phase 1 stopping rules, and measures the time until one suggestion is complete. Each prefix is sent once without the prompt cache and once with it (cut points of a message in order).

**Latency, all 240 messages (4,510 prefixes), median ms [90th percentile].**

| granularity | n | Hugging Face fp32 (Phase 1) | llama.cpp Q4_K_M, no cache | llama.cpp Q4_K_M, cache | median tokens |
|---|---|---|---|---|---|
| partial_word | 910 | 98 [169] | 45 [61] | 45 [71] | 2 |
| next_word | 1,200 | 92 [142] | 41 [50] | 38 [53] | 2 |
| phrase | 1,200 | 266 [395] | 68 [89] | 64 [88] | 5 |
| sentence | 1,200 | 553 [1527] | 145 [356] | 140 [364] | 14 |

**Q4_K_M against Q8_0, first 20 messages (383 prefixes), median ms with cache.**

| granularity | Q4_K_M | Q8_0 |
|---|---|---|
| partial_word | 38 | 45 |
| next_word | 36 | 40 |
| phrase | 56 | 75 |
| sentence | 137 | 158 |

**Frozen rule with the llama.cpp times** (`src/swap_latency.py` replaces only the latency in a copy of the Phase 1 results; `src/eval_frozen.py` as for the second-dataset check, `PHASE2A_POLICY_PLAN.md` §1.6).

| group | calls % | missed %, Phase 1 times | missed %, llama.cpp Q4_K_M times | value/call (always), Phase 1 → llama.cpp |
|---|---|---|---|---|
| sentence | 38.3 [34–42] | 31.1 [24–39] | 31.4 [25–39] | 0.25 (0.17) → 0.29 (0.20) |
| word-level mix | 54.1 [52–56] | 30.5 [25–36] | 30.8 [25–36] | 0.58 (0.40) → 0.58 (0.40) |
| partial_word | 38.7 [36–42] | 49.0 [41–57] | 49.3 [41–57] | 0.29 (0.23) → 0.30 (0.23) |

**Observations.**
- llama.cpp Q4_K_M is about 2 times faster for word-level suggestions and about 4 times faster for phrase and sentence.
- Q8_0 was not faster than Q4_K_M in the 20-message run, and the file is larger (767 MB against 462 MB). Q4_K_M is used from here on.
- The prompt cache changes the medians by a few milliseconds at most. The prefixes are short, so most of the time is generation.
- The policy numbers are nearly unchanged with the faster times (missed shares within 0.3 points). Value per call rises for sentence, for the rule and for always-invoke alike.
- Sanity check of the output: the share of suggestions whose first word equals the reference is close for both runtimes (next_word 23.9 % and 23.9 %, phrase 23.5 % and 23.3 %, sentence 24.2 % and 23.7 %; partial_word exact match 3.0 % and 3.0 %).

**Cautions.**
- The Phase 1 times come from an earlier run on the same machine, so the speed ratios are approximate.
- In the policy rerun, the suggestions and their scores are the Phase 1 ones; only the time was replaced.
- The first-word check uses one sampled suggestion per prefix. It is not the Phase 1 quality metric, and the quantized model was not scored with that metric.
- The cache run sends 5 cut points per message. Real typing extends the text one key at a time, so the cache effect in use was not measured.
- One machine (Apple Silicon laptop); no LiteRT measurement (Phase 3b).

## Reproduce

Start the server in another terminal first, with one GGUF file at a time:

```bash
llama-server -m <path>/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 -c 2048
```

Then, from the repo root:

```bash
uv run python -m src.bench_llama Q4_K_M                           # first 20 messages
uv run python -m src.bench_llama Q8_0                             # after restarting the server with the Q8_0 file
MESSAGES=240 uv run python -m src.bench_llama Q4_K_M_all          # all 240 messages, about 15 min
uv run python -m src.swap_latency Q4_K_M_all                      # Phase 1 results with llama.cpp times
RESULTS=phase1_results_v3_llama_Q4_K_M_all.csv uv run python -m src.eval_frozen   # frozen rule, llama.cpp times
```

## Files

| File | What it is |
|---|---|
| `src/bench_llama.py` | sends the Phase 1 prefixes to a llama.cpp server and measures the time per suggestion |
| `src/swap_latency.py` | copies the Phase 1 results with the llama.cpp latency |
| `phase1_results_v3_llama_Q4_K_M_all.csv` | Phase 1 results with llama.cpp Q4_K_M times |
| `results/llama_bench_Q4_K_M.txt`, `results/llama_bench_Q8_0.txt` | 20-message runs |
| `results/llama_bench_Q4_K_M_all.txt`, `results/llama_bench_Q4_K_M_all.csv` | 240-message run (table and one row per request) |
| `results/llama_bench_*_summary.csv` | medians per granularity |
| `results/eval_frozen_oasst2_llama_Q4.txt` | frozen rule with the llama.cpp times |
