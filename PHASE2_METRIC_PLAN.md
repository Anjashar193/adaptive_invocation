# Fixing the Usefulness Metric Before Phase 2

**One-line summary:** Your evaluation metric has never been validated against what it claims to measure. This plan builds a small control dataset that decides, per granularity, which metric Phase 2 should optimize — before the policy is written on top of it.

---

## Progress

- [x] **Stage 7** — Repo defect fixes *(independent, do early)*
  - [x] `src/__init__.py` + dual-style imports in `report.py` / `run_eval.py`
        *(note: `python src/run_eval.py` already worked — the broken form was
        the README's `from src.report import …`, now fixed)*
  - [x] Remove `recompute_metrics.py` from README (file does not exist)
  - [x] `N_MESSAGES` env-gated, default frozen at 100; `N_MESSAGES_FULL = 260`
  - [x] README: usefulness labelled 0–2 with full distributions (verified
        189/89/122 · 140/133/28 · 339/49/12 · 333/56/11); partial_word top-5
        corrected 0.150 → 0.046 (ref ≥3 chars)
- [x] **Stage 0** — Keystroke primitives in `src/metrics.py`
      *(`keystrokes_saved`, `keystroke_rate`, `chars_saved_per_second` + smoke tests;
      `score_example` untouched)*
- [x] **Stage 1** — `ks_rate` / `cs_per_sec` / `ks_zero` in `src/report.py`
      *(reproduces predicted 0.259/0.099/0.071/0.039 and 2.71/0.54/1.58/0.50;
      **verified** `usefulness`, `top5_acc` and all metric columns byte-identical)*
- [x] **Stage 2** — `src/label_join.py` (299/300 join + similarity cache)
      *(verified: 299/300 joined, only `id=6` lost, 8 ambiguous keys with
      **0** usefulness conflicts, all 299 sims reproduce the earlier analysis
      bit-for-bit; `assert_join_health` fails loudly on drift)*
- [x] **Stage 3** — `src/agreement.py` (κ, Spearman, paired bootstrap, ceiling, CV)
      *(reproduces from versioned code: all four verdicts, κ +0.300 / linear
      +0.400, bias 106-vs-16, exact 48.0/70.3/56.0/62.7, ceiling
      69.3/74.3/65.3/66.7 → 68.9% overall. CV now optimizes linear-weighted κ,
      so tuned accuracies differ slightly from the earlier draft — but the
      claim holds: **sentence still degrades**, 62.7% → 58.7%.)*
- [x] **Stage 4** — Fix `compare_label.py` + `hand_label.py` (3-pass rubric)
      *(`compare_label.py` runs clean — no `KeyError` — and prints agreement,
      κ, paired verdicts and the confusion matrix; `hand_label.py` rewritten
      for A1/A2 passes storing **full prefixes** + `result_row_id`;
      README calibration section updated with κ and the construct-mismatch
      finding)*
- [x] **Stage 5** — `src/control_set.py` (stratified sampling + HT weights)
      *(345 items at the power-derived sizes 120/75/75/75; all three strata
      populated; HT weights recover pool sizes within 4%; **verified every
      item within a granularity has a unique prefix** — the first pass left
      4–5 exact duplicates that would have broken bootstrap independence)*
- [x] `tests/` — metrics, label_join, agreement *(**36 passing**)*
- [ ] **— HUMAN LABELING —** ← **YOU ARE HERE.** 345 items in `control_set.csv`,
      3 passes (A1 plausible → A2 matches → A3 preference) + 100-item overlap
      block for a second labeler
- [ ] **Stage 6** — `src/decide_metric.py` → `metric_choice.json`
      *(blocked on labels; the decision rule is pre-registered in §3.2)*

**All code stages are complete.** Everything above the labeling line runs and is
verified — `uv run pytest` (36 tests) and `uv run python compare_label.py`
reproduce every statistic in this document from versioned code.

> ⚠ **Blocked on a decision before Stage 5:** `PROJECT_GUIDELINE.md` §2.0 mandates
> `usefulness_score` as Phase 2's primary metric. If this work retires it, the
> guideline needs an explicit amendment. See §4.3.

---

# Part 1 — The Problem

## 1.1 Where the project stands

```
Phase 1  Generation quality baseline        ████████████████████  DONE (verified)
Phase 2.0  Threshold calibration            ████████░░░░░░░░░░░░  DONE, but result is unusable
Phase 2a  When to invoke  ← the thesis      ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
Phase 2b  How much to generate              ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
Phase 3  Production runtime                 ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
Phase 4  Demo                               ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
```

Phase 1 is solid — every README number reproduces exactly from `phase1_results_v2.csv`. The contribution, Phase 2, has not begun.

## 1.2 The blocker

Everything in Phase 2 is scored by `metrics.usefulness_score()`, which thresholds embedding similarity at 0.75 / 0.40. Phase 2.0 hand-labeled 300 examples to check those thresholds and got **58.8% agreement**. The obvious reading is "thresholds need tuning." That reading is wrong.

**Finding 1 — the errors all point one way.**

```
auto scored TOO LOW   ████████████████████████████████████  106
auto scored TOO HIGH  █████                                  16
```

Random noise is symmetric. This is systematic bias.

**Finding 2 — the two raters are measuring different things.**

Actual cases your labeler marked "plausible" but the scorer marked "irrelevant":

| prefix continues… | model said | user actually typed | human | auto |
|---|---|---|---|---|
| …explain how to | `quantum` | `make` | 1 | 0 |
| …I was | `fire` | `I` | 1 | 0 |
| …good at | `C++` | `speaking` | 1 | 0 |

The human scored **"is this a valid thing to type here?"** (plausibility). The scorer measured **"is this close in meaning to what was actually typed?"** (reference-match). Both are reasonable. They are not the same question, and no threshold reconciles them.

**Finding 3 — proof the gap is real, not just calibration.**

Within cases where the prediction shares *zero* leading characters with the reference, the human score still spreads 39–45%, and embedding similarity explains almost none of it:

| Granularity | human still spreads | ρ(similarity, human) in that region |
|---|---|---|
| next_word | 44% | **+0.135** |
| partial_word | 25% | **−0.112** |
| phrase | 45% | +0.516 |
| sentence | 39% | +0.198 |

For the two short granularities, the current metric is blind to the axis the human is actually using.

**Finding 4 — tuning cannot fix it.** Best *possible* accuracy from similarity with thresholds cheated onto the labels: **68.9%** (vs. 55.2% majority-class). And honest 5-fold CV of tuning makes `sentence` *worse* (62.7% → 57.3%) and drops `partial_word` κ (0.323 → 0.256).

```
            current 58.8%   ──►   tuned ~65%   ──►   ceiling 68.9%
                                                      ↑
                                      no amount of tuning passes this
```

## 1.3 The problem behind the problem

> **We cannot currently tell whether the scorer is bad or the task is ambiguous.**

Simulation: if a second labeler agreed with the first at κ≈0.52 (70% raw agreement), then *no metric, however perfect, could exceed ~70% agreement with labeler 1.*

```
observed ceiling      68.9%  ──┐
noise ceiling if      70.0%  ──┘  ← statistically indistinguishable
human-human κ≈0.52
```

You have **one labeler and zero inter-annotator agreement**, so these two explanations cannot be separated. Until they are, the central claim of the Phase 2.0 chapter is unfalsifiable — and it is the first thing a reviewer will ask about.

## 1.4 Secondary defects

| Defect | Impact |
|---|---|
| `compare_label.py` crashes (`KeyError: 'usefulness_auto'`) | None of the reported agreement numbers reproduce |
| README references `recompute_metrics.py` — **file does not exist** | Documented pipeline is unrunnable |
| `run_eval.py` uses bare imports; no `src/__init__.py` | README commands fail from repo root |
| `N_MESSAGES=100` − 20% holdout ≈ 80 eval messages | README claims 200; guideline requires >200 |
| `usefulness 0.833` printed like a rate | It's a mean over {0,1,2}; true spread is 189/89/122 of 400 |
| README shows partial_word top-5 `0.150` | Honest figure is `0.046` for references ≥3 chars |

---

# Part 2 — The Solution

## 2.1 Core idea

Your own ROARS rule **A — Ascend the Supervision Ladder** says: Symbolic > Unsupervised > Self-Sup > **Supervised** > RL > LLM-judge. Hand-labeling is *Supervised*. The symbolic rung was skipped.

So test it. `keystrokes_saved(pred, ref)` = length of the correct leading character run. No labels, no thresholds, no embedding model, runs in under a second on data you already have.

**It reproduces your entire Phase 1 conclusion:**

| Granularity | Qwen KS-rate | Best baseline | ms/call | **chars saved / sec** |
|---|---|---|---|---|
| next_word | 25.9% | 5.2% | 413 | **2.7** |
| partial_word | 9.9% | 3.4% | 456 | 0.5 |
| phrase | 7.1% | 1.1% | 1098 | 1.6 |
| sentence | 3.9% | 0.7% | 3181 | 0.5 |

That last column is the cost-aware number a policy optimizes — and your current metric cannot produce it. The 5× gap between next_word and sentence **is** the argument for adaptive invocation, in units a policy can act on.

This reframes the control set's job:

> Not *"validate the embedding scorer"* but **"for which granularities can we drop human labels entirely?"**
> Every granularity answered *yes* costs zero labels forever after.

## 2.2 Which metric wins where

Paired bootstrap on the Spearman difference (`keystrokes − embedding`), 4000 resamples, same items scored by both:

```
partial_word   ├──●──────────┤        +0.263  [+0.042, +0.479]   KEYSTROKES WIN
                    │
next_word      ├────●────┤            +0.040  [−0.099, +0.190]   tie
                    │
sentence    ├──────●────┤             −0.162  [−0.417, +0.085]   tie
                    │
phrase   ├──────●───┤                 −0.331  [−0.553, −0.103]   EMBEDDINGS WIN
                    │
              ──────0──────  ← CI crossing zero = not separable
```

**Verified seed-stable** (seeds 99 and 12345 give identical verdicts). Only 2 of 4 separate — and that is an honest finding, not a failure.

**Why not just combine them?** Naive `max(ks_rate, sim)` is *worse* than the better single metric everywhere: next_word .599 vs .620 · partial_word .208 vs .484 · phrase .458 vs .496 · sentence .318 vs .331. Combination needs design, not an OR.

**Why keystrokes can't win everywhere** — it is zero for most items, and humans still credit many of those (valid paraphrases save no literal characters):

| | next_word | partial_word | phrase | sentence |
|---|---|---|---|---|
| keystrokes == 0 | 64% | 82% | 63% | 72% |

## 2.3 The rubric — three axes, none redundant

Each was tested against the data before being included.

```
                    ┌─────────────────────────────────────────────┐
  A1  PLAUSIBLE     │ reference HIDDEN — prefix only              │  0/1
      "could a writer continue this way?"                         │
                    ├─────────────────────────────────────────────┤
  A2  MATCHES       │ reference SHOWN                             │  0/1/2
      "does it agree with what was actually typed?"               │
                    ├─────────────────────────────────────────────┤
  A3  PREFERENCE    │ two candidates, same prefix — which is better?│ ordinal
                    └─────────────────────────────────────────────┘
```

**Why not one wider scale (0–4, or a slider)?** Bucketing items by `(similarity, keystrokes)` leaves **24–36% irreducible disagreement inside cells** — identical metric values, different human scores. The missing information isn't in the features, so widening the scale cannot recover it.

**Why not preference alone?** The base rate of "worth sending" swings wildly, and a pure ranking has no zero point:

```
partial_word  34%  ████████
sentence      43%  ██████████
phrase        51%  ████████████
next_word     88%  ██████████████████████
```

A policy needs an absolute send / don't-send cut. Preference is necessary but not sufficient.

**A1 must be a separate pass from A2** — if the reference is visible it anchors the plausibility judgment and the axis separation is fictional.

## 2.4 How big must the control set be?

Not a guess — simulated by subsampling the existing labels and re-running the actual decision, 60 trials per size:

| Granularity | truth | n=40 | n=60 | n=75 |
|---|---|---|---|---|
| partial_word | keystrokes win | 37% | **58%** ⚠ | — |
| phrase | embeddings win | 52% | 88% | 100% |
| next_word | tie | 100% | 98% | 100% |
| sentence | tie | 95% | 98% | 100% |

*(for ties, the % is the rate of correctly refraining from declaring a winner)*

**`partial_word` at n=60 detects its own true effect 58% of the time — a coin flip.** It is the binding constraint. Extrapolated: n=90 → 75%, **n=120 → 85%**, n=150 → 87%.

The ties show ≥95% specificity everywhere, so the design will not manufacture false winners. All the risk is on the power side.

> **Final sizing: 120 for `partial_word`, 75 each for the rest = 345 items.**

Stratifying on the `keystrokes == 0` boundary is worth doing (it's free) but the gain is modest — re-verification put it at **1.04–1.12×**, not the ~1.3× an earlier coarse comparison suggested. Size is set by power, not by stratification.

## 2.5 Sampling design

Three strata per granularity, because uniform sampling under-represents exactly the cases that decide the question:

| Stratum | Share | What it is | Why |
|---|---|---|---|
| **S1** disagreement | ~40% | highest `\|rank(sim) − rank(ks)\|` | where the decision is actually made |
| **S2** paraphrase mass | ~35% | `ks == 0` but high similarity | decides if symbolic is admissible on long granularities (21/47 phrase items here got human credit) |
| **S3** anchor | ~25% | uniform random | estimates and corrects stratification bias |

⚠ **Store per-item inclusion probabilities and Horvitz-Thompson reweight all reported rates back to the uniform population.** Stratified sampling biases headline numbers *by construction*; the correction belongs in code, not in the discussion section.

Reuse `hand_label.dedupe_by_source_message` — the guideline's rule that all 5 cut points of a message stay on one side of a split applies here too, or items aren't independent and the CIs are falsely narrow.

## 2.6 Inter-annotator agreement — required, not optional

**100 items (25 per granularity) double-labeled.** This is what makes §1.3 resolvable. Report per-axis κ:

- A1 and A3 should reach **κ ≥ 0.6**
- A2 will be lower — **and that is itself a finding.** A low A2 κ means the 3-class scale is not a reproducible target, and A3 preference becomes primary.

---

# Part 3 — Implementation Guideline

## 3.1 Dependency order

```
  Stage 0  primitives (metrics.py)         no deps
  Stage 7  repo defect fixes               independent ── do early, unblocks reproducibility
     │
     ├──► Stage 1  report integration
     └──► Stage 2  label↔result join
                │
                └──► Stage 3  agreement statistics
                        │
                        ├──► Stage 4  fix compare_label.py / hand_label.py
                        └──► Stage 5  control set  ── power analysis BEFORE labeling
                                 │
                                 └──► [ HUMAN LABELING  ~3 passes + overlap block ]
                                          │
                                          └──► Stage 6  the decision
```

Stages 0–4 and 7 are pure code and reproduce every number in Part 1 and Part 2 from versioned source.

## 3.2 Stage-by-stage

### Stage 0 — Primitives · `src/metrics.py` (append only)

```python
keystrokes_saved(prediction, reference) -> int      # correct leading run, raw chars
keystroke_rate(prediction, reference)   -> float    # saved / len(reference), 0.0 if empty
chars_saved_per_second(keystrokes, latency_ms)      # guard latency_ms <= 0
```

Reuse `normalize` / `trim_prediction`. Extend the existing `__main__` smoke block (repo convention — 5 files do this).

> ⚠ **Do not call these from `score_example()`.** That changes the CSV schema and risks the Phase 1 reproduction. Derive them at report time from `prediction` + `reference`, already in the CSV. *(ROARS: Respect the Baseline.)*

### Stage 1 — Report · `src/report.py`

Add `ks_rate`, `cs_per_sec`, `ks_zero_frac` columns via Stage 0. Reuse `load_results` / `_avg` unchanged.

- `ks_zero_frac` (63–82%) is the symbolic metric's headline limitation — keep it visible in every table.
- ⚠ **`ks_rate` is not comparable across granularities.** Median reference length is 4 chars (next_word, partial_word) vs 25 (phrase) and 35 (sentence) — a ~9× spread. Only `cs_per_sec` is cross-comparable, since it normalizes by compute rather than target length. Say so next to the table.

**Gate:** `usefulness` and `top5_acc` columns byte-identical before/after.

### Stage 2 — Join · `src/label_join.py` *(new)*

Replaces throwaway scripts. The naive `"|"` split recovers only **259/300** — phrase/sentence candidates contain literal `|` and `\r\n`.

```python
normalize_candidates_from_label(s)    # split on r'(?:^|\s\|\s)\d+\.\s'  → recovers 299/300
normalize_candidates_from_results(s)  # split on " | "
join_labels_to_results(labels, results)   # filter system=="qwen"; return match_count
attach_similarity(joined, cache_path)     # see below
```

- The single unrecoverable row is `id=6` (partial_word, ref `20`).
- **8 keys map to multiple rows, but zero have conflicting `usefulness`** — verified, so all reported stats are unaffected. Still assert single-match-or-documented-collapse; never silently take `[0]`.
- `semantic_sim` is **absent from the CSV for partial_word and next_word** — recompute and cache by `(granularity, prediction, reference)`. This is the plan's only real compute cost.
- Hard-assert ≥299/300 so future schema drift fails loudly instead of quietly changing thesis numbers.

### Stage 3 — Statistics · `src/agreement.py` *(new)*

`scipy` and `sklearn` are already importable transitively — no new dependencies.

```python
confusion_matrix_3x3, agreement_rates          # exact / off-by-one / off-by-two
kappa(...)                                      # unweighted AND weights="linear"
spearman(...)
paired_bootstrap_spearman_delta(human, a, b, n=4000, seed)
oracle_threshold_ceiling(...)  /  cv_threshold_tuning(k=5)
```

- **Report linear-weighted κ.** With 36.8% off-by-one vs 4.4% off-by-two errors, weighting is the correct choice and unweighted under-reports your scorer: overall **κ 0.300 → 0.400**; next_word 0.257 → 0.352.
- ⚠ In the bootstrap, **resample item indices once per replicate and score both metrics on the same resample.** This pairing is the entire point; it's the easiest thing to get wrong.
- Seed every bootstrap and CV split. Unseeded CIs are not publishable.

### Stage 4 — Fix the labeling tools

- **`compare_label.py`** — rewrite as a thin CLI over Stages 2–3. Print per granularity: n, exact %, unweighted + linear-weighted κ, Spearman, paired bootstrap Δ with CI.
- **`hand_label.py`** — write the current schema plus a **full untruncated `prefix`** and a stable `result_row_id`. The `prefix[-100:]` truncation is exactly why the join is lossy and ambiguous; do not reintroduce it. Add the three-pass A1/A2/A3 flow and the double-labeled overlap block.

### Stage 5 — Control set · `src/control_set.py` *(new)*

Implements §2.4 and §2.5. Run the power simulation on existing labels **before labeling anything** — the decision to label at all is itself a cheap experiment *(ROARS: Run Cheap Experiments)*.

> ⚠ The 299 existing labels **must be re-labeled** under the new axes. A single old score cannot be decomposed into A1/A2 after the fact.

### Stage 6 — Decision · `src/decide_metric.py` *(new)*

Pre-registered, per granularity, against A3 preference as ground truth:

1. **CI excludes 0** → adopt the winner. *(Expected: partial_word → keystrokes; phrase → embeddings.)*
2. **CI includes 0** → **do not tune, do not combine.** Adopt **keystrokes** — simpler, label-free, threshold-free — and report the tie with its CI.
   - A tie is a legitimate result. Breaking it by simplicity is principled *(ROARS: Optimize for Simplicity)*; breaking it by point estimate is p-hacking.
   - Evidence against combining: naive combination is worse everywhere (§2.2). Against tuning: honest CV *lowers* sentence accuracy and partial_word κ (§1.2).
3. Emit versioned `metric_choice.json` → Phase 2 imports it and must not re-litigate.

### Stage 7 — Repo defects *(independent, do early)*

- Add `src/__init__.py`; fix `run_eval.py` bare imports.
- Delete the `recompute_metrics.py` line from the README — the file doesn't exist. Pipeline is `run_eval.py` → `report.py`.
- Raise `N_MESSAGES` to 260 (>200 eval messages after holdout) — but **gate behind a flag and keep `phase1_results_v2.csv` frozen.** Do not re-run before Stage 6 concludes.
- README: report usefulness as a distribution (`next_word: 189/89/122 of 400`) alongside the mean; replace partial_word top-5 `0.150` with the honest `0.046 (ref ≥3 chars)`.
- Add `tests/`: `test_metrics.py`, `test_label_join.py` (299/300 + the 8-ambiguity case), `test_agreement.py` (bootstrap determinism under fixed seed).

## 3.3 Verification

| # | Check | Expected |
|---|---|---|
| 1 | `pytest tests/` | green |
| 2 | `report.py` on frozen CSV | `usefulness`/`top5_acc` byte-identical; `ks_rate` = 25.9/9.9/7.1/3.9; `cs_per_sec` = 2.7/0.5/1.6/0.5 |
| 3 | `compare_label.py` | no `KeyError`; 299/300 joined; exact 58.8%; κ +0.300 / weighted +0.400; per-gran 48.0/70.3/56.0/62.7 |
| 4 | `agreement.py` paired CIs | partial_word +0.263 [+0.042,+0.479]; phrase −0.331 [−0.553,−0.103] |
| 5 | Seed stability | verdicts identical at seeds 99 and 12345 |
| 6 | README commands | all copy-paste and run from repo root |
| 7 | Post-labeling power re-check | `partial_word` ≥80%; if not, extend that granularity before deciding |

---

# Part 4 — What You Get, and What's Next

## 4.1 Outcomes

| # | Outcome | Why it matters |
|---|---|---|
| 1 | A defensible per-granularity metric | Replaces one number that disagrees with your labeler 41% of the time |
| 2 | Some granularities need **zero** labels forever | Keystrokes is arithmetic — no model, no thresholds, no labeling |
| 3 | You learn whether the scorer is *actually* bad | Resolves §1.3 — currently unfalsifiable, and a reviewer's first question |
| 4 | Every number reproducible from versioned code | Today `compare_label.py` crashes and the stats live in throwaway scripts |
| 5 | Phase 2 gets a metric it can optimize | `chars/sec`: next_word 2.7 vs sentence 0.5 — the 5× gap *is* the thesis motivation |

## 4.2 Honest cost

345 items across three passes, of which the 299 existing labels must be redone under the new axes. That is real rework.

The case for doing it anyway is your own guideline:

> *"Phase 2's success is measured with these thresholds — if they're wrong, everything after is wrong."*

Finding out at the defense costs more than finding out now.

## 4.3 What to do next

**Immediately (no decisions needed, ~half a day):** Stage 7 + Stage 0. Independent of everything, restores reproducibility, adds the symbolic metric without touching Phase 1 numbers.

**Then (~2 days):** Stages 1–4. At this point every statistic in this document is versioned, tested, and reproducible from the README.

**Then — one decision is yours, not mine:**

> `PROJECT_GUIDELINE.md` §2.0 mandates `usefulness_score` as Phase 2's primary metric. If this work retires it in favour of preference/keystrokes, **the guideline needs an explicit amendment** — otherwise Phase 2 gets built on the metric this work just deprecated. Resolve before Stage 5.

**Then (~1 week, mostly calendar time):** Stage 5 power analysis → labeling → Stage 6 decision → `metric_choice.json`.

**Then: Phase 2a — the actual contribution.** Typing simulation from AmazonQAC, then always-invoke → fixed-interval → pause-based → boundary-based → learned.

## 4.4 Open issues

- **`partial_word` class "2" has n=3.** No per-class statistic there is meaningful — report binary or ordinal only. Binarizing isn't a free fix either: it drops next_word κ to +0.061 through class imbalance (the human said 0 only 9/75 times).
- **`cs_per_sec` rests on MPS float32 latency**, which the guideline already flags as non-production. Any policy optimizing it will re-rank once Phase 3 swaps in GGUF/llama.cpp with KV caching. Report as a ratio, re-derive in Phase 3, never hard-code the constants into the policy.
- **Multilingual** (raised at the interim presentation): adding languages is breadth, not novelty, and only Spanish has usable volume in OASST2 validation (es 410, de 121, fr 70 prompter messages ≥6 words). The defensible framing is a **tokenization-latency stress test** — do per-language token costs shift the optimal invocation threshold? — run on Spanish *after* Phase 2 works in English. Do not replicate a broken measurement in two languages.
