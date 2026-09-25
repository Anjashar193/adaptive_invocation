# Adaptive Invocation for Autocomplete Generation

**Question:** Can a lightweight adaptive invocation policy cut unnecessary model
calls without losing useful suggestions?

```
Phase 1   Generation quality baseline        DONE
Phase 2.0 Metric validation & calibration    ONGOING
Phase 2   Invocation policy (the thesis)     NOT STARTED  <- next
Phase 3   Production runtime validation      NOT STARTED
Phase 4   Demo                               NOT STARTED
```

---

## Dataset

| | |
|---|---|
| Source | [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2), English `prompter` messages, validation split |
| Method | Each message cut at 15/30/50/70/85% to simulate mid-typing |
| Size | 500-800 examples per granularity, from 200 messages |

## Phase 1 — Generation Quality Baseline- DONE

| Granularity | Qwen usefulness (0–2) | dist 0/1/2 | Best baseline | Top-5 accuracy | Latency |
|---|---|---|---|---|---|
| next_word | 0.833 | 189/89/122 | 0.435 | 0.420 | 413ms |
| partial_word | 0.628 | 140/133/28 | 0.535 | 0.046 † | 456ms |
| phrase | 0.182 | 339/49/12 | 0.010 | 0.323 | 1,098ms |
| sentence | 0.195 | 333/56/11 | 0.000 | 0.278 | 3,181ms |

Qwen beats the baseline at every granularity. Gap is largest at
phrase/sentence (15-20x). Latency jumps sharply with granularity, which is
the reason invocation policy matters (Phase 2): don't pay sentence-level
cost on every keystroke.

---

## Phase 2.0 — Metric Validation & Calibration- Ongoing

### Round 1: initial calibration (300 examples, single labeler)

300 examples hand-labeled (75 per granularity, deduplicated by source
message), scored 0/1/2, and compared against the automatic scorer's
0.75/0.4 similarity thresholds.

| Granularity | Examples | Agreement | κ | κ (linear) |
|---|---|---|---|---|
| Partial word | 74 | 70.3% | +0.323 | +0.423 |
| Sentence | 75 | 62.7% | +0.201 | +0.249 |
| Phrase | 75 | 56.0% | +0.149 | +0.177 |
| Next word | 75 | 48.0% | +0.257 | +0.352 |
| **Overall** | **299** | **59.2%** | **+0.300** | **+0.400** |

| Agreement between automatic and manual scores | Count | % |
|---|---|---|
| Exact match | 177 | 59.2% |
| Off by 1 point | 109 | 36.5% |
| Off by 2 points | 13 | 4.3% |

Disagreement direction: the automatic scorer was lower than the human 106
times, higher 16 times.

This round used one labeler; no inter-annotator agreement was measured.

### Round 2: control set and two-axis labeling

A control set of 345 items was built (`src/control_set.py`): 120
partial_word, 75 each of next_word/phrase/sentence. 

Scoring was split into two separately-run passes:
- `plausible` (reference hidden): scored 0/1
- `matches` (reference shown): scored 0/1/2

All 345 items were labeled by one labeler (Labeler A) on both axes, in that
order.

A second labeler (Labeler B) was assigned 100 items. A proportional sample
(35/22/22/21) taken from within the existing 345 items, duplicated under new
item IDs and shuffled across granularities, so the second labeler's batch
spans all four granularities. Labeler B labeled this corrected batch on both
axes.

### Metric decision

Two candidate metrics were compared: `semantic_similarity` (the original
embedding-based metric) and `keystrokes_saved`.

For each granularity, a paired bootstrap compared each metric's correlation with the `matches`-axis human scores. Decision rule, fixed before results were computed: if the resulting confidence interval excludes zero, adopt the higher-correlation metric; if it includes zero, adopt `keystrokes_saved`.

| Granularity | keystrokes_saved − embeddings, 95% CI | Result | Metric adopted |
|---|---|---|---|
| next_word | delta -0.078, [-0.250, +0.075] | CI includes 0 | keystrokes_saved |
| partial_word | delta +0.244, [+0.047, +0.448] | CI excludes 0 | keystrokes_saved |
| phrase | delta -0.150, [-0.426, +0.139] | CI includes 0 | keystrokes_saved |
| sentence | delta -0.386, [-0.646, -0.122] | CI excludes 0 | semantic_similarity |

The CI is the range where the true difference between the
two metrics most likely falls. If that range crosses zero, both a real
advantage and no advantage at all are still possible, too uncertain to
call.

| Granularity | Does the range cross zero? | What that means |
|---|---|---|
| next_word | Yes (-0.250 to +0.075) | Can't tell which metric is better |
| partial_word | No (+0.047 to +0.448, all positive) | keystrokes is consistently ahead |
| phrase | Yes (-0.426 to +0.139) | Can't tell which metric is better |
| sentence | No (-0.646 to -0.122, all negative) | embeddings is consistently ahead |

Recorded in `metric_choice.json` (version 1.0, generated 2026-09-25T07:32:18Z,
source `control_labels.csv`, MD5 `ba25ddfe1319e95f3814fbb37f41136c`,
890 source rows).

### Inter-annotator agreement

Labeler A's and Labeler B's scores were compared on the 100 overlapping
items, matched by `(granularity, prefix, reference)`.

| Axis | Granularity | n | Exact agreement | κ |
|---|---|---|---|---|
| plausible | next_word | 22 | 95.5% | 0.879 |
| plausible | partial_word | 35 | 94.3% | 0.839 |
| plausible | phrase | 22 | 90.9% | -0.048 |
| plausible | sentence | 21 | 95.2% | 0.644 |
| matches | next_word | 22 | 68.2% | 0.505 |
| matches | partial_word | 35 | 94.3% | 0.692 |
| matches | phrase | 22 | 54.5% | 0.027 |
| matches | sentence | 21 | 71.4% | 0.447 |

> [!NOTE]
> One thing to notice here: κ for 'plausible' phrase is -0.048.

### Post-decision checks

- `uv run pytest`: 36 tests passed
- Bootstrap seed stability: verdicts identical at seed 99 and seed 12345,
  for all four granularities
- Power re-check (bootstrap resampling the final labeled data, 60 trials,
  checking how often the resampled verdict matches the real verdict):
  next_word 80.0%, partial_word 68.3%, phrase 80.0%, sentence 86.7%
- `partial_word`'s `matches`-axis score distribution: 0 → 139, 1 → 12, 2 → 4
  (n=155). 


##  Gaps
partial_word's metric decision has below-target power (68.3% against an 80% target)
partial_word's matches-axis score distribution is imbalanced: 0 → 139, 1 → 12, 2 → 4 (n=155) — very likely the direct cause of the power shortfall above.

## Next (per PROJECT_GUIDELINE.md, 2a)

## Run It

Set up and reproduce Phase 1:
```bash
uv sync
uv run python src/run_eval.py
uv run python -c "from src.report import load_results, summarize; summarize(load_results('phase1_results_v2.csv'))"
```

Reproduce the Round 1 calibration:
```bash
uv run python compare_label.py
uv run pytest
```

Build the control set and reproduce the Round 2 metric decision:
```bash
uv run python -m src.control_set        # writes control_set.csv (345 items)
uv run python hand_label.py             # labeling instructions
uv run python -m src.decide_metric      # writes metric_choice.json
uv run python -m src.inter_annotator    # inter-annotator agreement, matched pairs
uv run python -m src.power_recheck      # post-labeling power check
```

---

## References

- ChaI-TeA (2025).
- Gmail Smart Compose (2019).
- Sequential Decision-Making for Inline Text Autocomplete (2024).