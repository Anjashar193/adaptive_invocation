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
| `report.py` | Prints summary table |

## The bug

| | |
|---|---|
| **Bug 1:** | Exact-match scoring returned 0% everywhere |
| Cause 1 | Exact-match too strict for open-ended text (also noted in ChaI-TeA, 2025) → switched to first-word match, char overlap, semantic similarity |
| Cause 2 | Usefulness score compared full model output to a short reference, unfairly penalizing longer predictions → fixed by trimming prediction to reference length first |
| Effect | Same predictions, corrected grading — numbers below are post-fix |
**Bug 2: generation** | Fixed token budgets caused predictions to overshoot their target (e.g. "error or issue in the" instead of "error"). Fixed with a custom stopping condition that halts generation at the actual word/phrase/sentence boundary. |

Same underlying model, corrected measurement — numbers below are post-fix.

## Results

Latency measured with PyTorch float32 on Apple M2 Max (MPS). Not production-representative — see Phase 3.

| Granularity | Qwen usefulness | Best baseline | Top-5 accuracy | Latency |
|---|---|---|---|---|
| next_word | 0.833 | 0.435 | 0.420 | 413ms |
| partial_word | 0.628 | 0.535 | 0.150 | 456ms |
| phrase | 0.182 | 0.010 | 0.323 | 1,098ms |
| sentence | 0.195 | 0.000 | 0.278 | 3,181ms |

Qwen beats the baseline at every granularity. Gap is largest at phrase/sentence
(15-20x), but absolute quality there is still low — longer completions aren't
solved yet. Latency jumps sharply with granularity, which is the whole reason
invocation policy matters (Phase 2/3): don't pay sentence-level cost on every
keystroke.

## Calibration (Phase 2.0)

300 examples hand-labeled (75 per granularity, deduplicated by source message)
and compared against the automatic scorer's 0.75/0.4 thresholds.

| Granularity | Examples | Agreement |
|---|---|---|
| Partial word | 71 | 69.0% |
| Sentence | 75 | 62.7% |
| Phrase | 75 | 56.0% |
| Next word | 75 | 48.0% |
| **Overall** | **296** | **58.8%** |

| Difference from hand label | Count | % |
|---|---|---|
| Exact match | 174 | 58.8% |
| Off by 1 | 109 | 36.8% |
| Off by 2 | 13 | 4.4% |

Open question: for next-word specifically,
should "plausible" mean grammatically valid in context, or semantically
related to the intended word — this choice swings next-word's agreement
significantly either way.

## Run it

```bash
uv sync
uv run python src/run_eval.py
uv run python -c "from src.recompute_metrics import recompute; recompute('phase1_results.csv', 'phase1_results_v2.csv')"
uv run python -c "from src.report import load_results, summarize; summarize(load_results('phase1_results_v2.csv'))"
```
To run calibration:

```bash
uv run python hand_label.py
uv run python compare_label.py
```

## Still missing / next

| Gap | Next step |
|---|---|
| Interpretation of "plausible" for next-word is ambiguous | re-check agreement |
| Single-reference eval undercounts valid completions | Known limitation, shared with ChaI-TeA |
| No typing simulation yet | Build one (pauses, timing) using AmazonQAC |
| No invocation policy yet | Compare always/fixed-interval/pause/boundary/adaptive |

## References

- ChaI-TeA (2025).
- Gmail Smart Compose (2019).
- Sequential Decision-Making for Inline Text Autocomplete (2024).
