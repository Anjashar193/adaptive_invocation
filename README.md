# Adaptive Invocation for Autocomplete Generation

Studies when a small language model (Qwen3-0.6B) should generate an autocomplete
suggestion while someone types and how much text to generate partial word,
next word, phrase or sentence under a limited latency budget.

**Question:** Can a lightweight adaptive invocation policy cut unnecessary model
calls without losing useful suggestions?

This repo = **Phase 1**: checking generation quality first. Invocation policy
itself is Phase 2/3.

## Dataset

| | |
|---|---|
| Source | [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2), English `prompter` messages, validation split |
| Method | Each message cut at 15/30/50/70/85% to simulate mid-typing |
| Size | 597-800 examples per granularity, from 200 messages |

## Pipeline

| File | Does |
|---|---|
| `data_prep.py` | Builds prefix/reference pairs from OASST2 |
| `baselines.py` | Frequency, bigram, random baselines |
| `metrics.py` | Scoring logic |
| `run_eval.py` | Runs Qwen3-0.6B + baselines, saves CSV |
| `recompute_metrics.py` | Re-scores existing CSV without re-running the model |
| `report.py` | Prints summary table |

## The bug

| | |
|---|---|
| Symptom | Exact-match scoring returned 0% everywhere |
| Cause 1 | Exact-match too strict for open-ended text (also noted in ChaI-TeA, 2025) → switched to first-word match, char overlap, semantic similarity |
| Cause 2 | Usefulness score compared full model output to a short reference, unfairly penalizing longer predictions → fixed by trimming prediction to reference length first |
| Effect | Same predictions, corrected grading — numbers below are post-fix |

## Results

Latency measured with PyTorch float32 on Apple M2 Max (MPS). Not production-representative — see Phase 3.

| Granularity | Qwen usefulness | Best baseline | Latency (MPS) |
|---|---|---|---|
| next_word | 0.806 | 0.456 | 108ms |
| partial_word | 0.568 | 0.447 | 115ms |
| phrase | 0.211 | 0.011 | 313ms |
| sentence | 0.172 | 0.014 | 800ms |

Qwen beats the baseline at every granularity. Gap is largest at phrase/sentence
(15-20x), but absolute quality there is still low — longer completions aren't
solved yet. Latency jumps sharply with granularity, which is the whole reason
invocation policy matters (Phase 2/3): don't pay sentence-level cost on every
keystroke.

## Run it

```bash
uv sync
uv run python src/run_eval.py
uv run python -c "from src.recompute_metrics import recompute; recompute('phase1_results.csv', 'phase1_results_v2.csv')"
uv run python -c "from src.report import load_results, summarize; summarize(load_results('phase1_results_v2.csv'))"
```

## Still missing / next

| Gap | Next step |
|---|---|
| Usefulness thresholds (0.75/0.4) are estimates | Check against hand-labeled examples |
| Single-reference eval undercounts valid completions | Known limitation, shared with ChaI-TeA |
| No typing simulation yet | Build one (pauses, timing) |
| No invocation policy yet | Compare always/fixed-interval/pause/boundary/adaptive |

## References

- ChaI-TeA (2025).
- Gmail Smart Compose (2019).
- Sequential Decision-Making for Inline Text Autocomplete (2024).