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
| Size | 500-800 examples per granularity, from 200 messages |

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
| Effect | Same predictions, corrected grading -numbers below are post-fix |
**Bug 2: generation** | Fixed token budgets caused predictions to overshoot their target (e.g. "error or issue in the" instead of "error"). Fixed with a custom stopping condition that halts generation at the actual word/phrase/sentence boundary. |

Same underlying model, corrected measurement , numbers below are post-fix.

## Results

Latency measured with PyTorch float32 on Apple M2 Max (MPS). Not production-representative. see Phase 3.

Usefulness is a **mean over a 0/1/2 scale**, not a percentage — 0.833 means
"averages between plausible and relevant", not "83% correct". The raw
distribution is given alongside so the mean can't be misread.

| Granularity | Qwen usefulness (0–2) | dist 0/1/2 | Best baseline | Top-5 accuracy | Latency |
|---|---|---|---|---|---|
| next_word | 0.833 | 189/89/122 | 0.435 | 0.420 | 413ms |
| partial_word | 0.628 | 140/133/28 | 0.535 | 0.046 † | 456ms |
| phrase | 0.182 | 339/49/12 | 0.010 | 0.323 | 1,098ms |
| sentence | 0.195 | 333/56/11 | 0.000 | 0.278 | 3,181ms |

† partial_word top-5 is **0.046** once restricted to references ≥3 characters
(n=109). The unrestricted 0.150 is inflated by one- and two-character
remainders that match coincidentally; `report.py` computes both.

Qwen beats the baseline at every granularity. Gap is largest at phrase/sentence
(15-20x), but absolute quality there is still low, longer completions aren't
solved yet. Latency jumps sharply with granularity, which is the whole reason
invocation policy matters (Phase 2/3): don't pay sentence-level cost on every
keystroke.

## Calibration (Phase 2.0)

300 examples hand-labeled (75 per granularity, deduplicated by source message)
and compared against the automatic scorer's 0.75/0.4 thresholds.

| Granularity | Examples | Agreement | κ | κ (linear) |
|---|---|---|---|---|
| Partial word | 74 | 70.3% | +0.323 | +0.423 |
| Sentence | 75 | 62.7% | +0.201 | +0.249 |
| Phrase | 75 | 56.0% | +0.149 | +0.177 |
| Next word | 75 | 48.0% | +0.257 | +0.352 |
| **Overall** | **299** | **59.2%** | **+0.300** | **+0.400** |

| Agreement between automatic and manual scores | Count | % |
|---|---|---|
| Exact match — same score (0, 1, or 2) | 177 | 59.2% |
| Minor disagreement — 1 point apart | 109 | 36.5% |
| Major disagreement — 2 points apart | 13 | 4.3% |

Report **linear-weighted κ** as the headline: since disagreement is
overwhelmingly one point apart, unweighted κ penalises "plausible vs relevant"
as harshly as a complete reversal and understates the scorer.

**Key finding 1 — disagreement is minor, not catastrophic.** The scorer rarely
gets things backwards, but it frequently draws the "plausible vs. relevant"
line in the wrong place.

**Key finding 2 — the error is systematic, not noise.** The scorer was too
*low* 106 times and too *high* only 16. Random noise would be symmetric.

**Key finding 3 — this is a construct mismatch, not miscalibration.** Cases
like predicting `quantum` where the user typed `make` were labelled "plausible"
by the human and "irrelevant" by the scorer: the human scored *grammatical
plausibility*, the scorer measures *similarity to the reference*. No threshold
reconciles the two. Best possible accuracy from similarity with thresholds
fitted directly on the labels is **68.9%** (vs. 55.2% majority-class), and
honest 5-fold CV of tuning makes `sentence` **worse** (62.7% → 58.7%).

**Open limitation:** one labeller, so there is no inter-annotator agreement.
The 68.9% ceiling is currently indistinguishable from an annotator-noise
ceiling — if two humans agree only ~70% of the time, no metric can do better.
See `PHASE2_METRIC_PLAN.md`.


## Run it

```bash
uv sync
uv run python src/run_eval.py
uv run python -c "from src.report import load_results, summarize; summarize(load_results('phase1_results_v2.csv'))"
```
To reproduce the calibration analysis:

```bash
uv run python compare_label.py       # agreement, kappa, paired metric comparison
uv run pytest                        # 36 tests
```

To build the control set for the next labelling round (see
`PHASE2_METRIC_PLAN.md`):

```bash
uv run python -m src.control_set     # writes control_set.csv (345 items)
uv run python hand_label.py          # labelling instructions
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
