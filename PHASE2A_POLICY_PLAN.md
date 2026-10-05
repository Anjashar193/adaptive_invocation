# Phase 2a — When to Invoke the Model

**One-line summary:** Phase 2a compares rules for *when* to call the model while a user types. A working pilot scores seven rules on all four numbers the research plan requires, using real typing pace (AmazonQAC) and a real gap shape (Aalto). It still runs on only about 80 messages and has two design limits (§4.2, §4.4), so no result here is final yet.

---

## Progress

- [x] **Stage A0** — AmazonQAC timestamp check *(answer: timestamps are per query, not per prefix or keystroke — see §1.1)*
- [x] **Stage A1** — Pace table from AmazonQAC: `make_pace_table.py` → `amazonqac_pace_quantiles.csv`
      *(210,000 train rows read from 3 of 20 files, 196,536 kept after trimming; median 0.67 s per typed prefix)*
- [x] **Stage A2** — `src/typing_sim.py` *(real pace per typist; gaps inside a query are modelled, see §2.1)*
- [x] **Stage A3** — Timing-only checks: `src/latency_vs_gaps.py`, `src/policy_timing.py`
- [x] **Stage A4** — Pilot: `src/policy_eval.py` *(7 rules, four numbers, staleness enforced in code)*
      *(gate passed: keystroke rate with zero delay reproduces the metric plan's 9.9 / 25.9 / 7.1 / 3.9 % exactly)*
- [x] **Stage A5** — Real gap shape from the Aalto 136M Keystrokes dataset: `fit_aalto_gaps.py` → `aalto_gap_shape.csv`, `src/gap_shape.py`, `policy_eval aalto`
      *(500 of 168,595 participant files, 7,452 sentences kept; word-boundary effect 1.48×, spread 0.58 — milder than the modelled styles, see §2.1)*
- [ ] **Stage A6** — More than 200 messages, with `message_id` saved and the split done by message *(needs `run_eval.py` changes; write to a new file)* ← **YOU ARE HERE**
- [ ] **Stage A7** — Learned policy *(only if a simple rule leaves a gap)*
- [ ] **Stage A8** — One run on AESLC
- [ ] `tests/` for `typing_sim` and `policy_eval`

> ⚠ **Everything in §1.3 is a pilot result.** The pilot rows are only the 80 messages already in `phase1_results_v2.csv`, the cut points sit at fixed positions (§4.2), and Aalto is copy-typing (§4.7). Do not cite these numbers in the write-up.

---

# Part 1 — The Problem and What the Pilot Shows

## 1.1 What AmazonQAC can and cannot tell us

The research plan says not to invent pause timings and to confirm whether AmazonQAC timestamps are per keystroke or per prefix. Checked directly on the downloaded files:

| Split | One row is | Timing fields |
|---|---|---|
| train | one query, with the ordered list of typed prefixes (backspaces included) | one start time (`first_prefix_typed_time`, millisecond precision on most rows) and one end time (`search_time`, whole seconds) |
| test | one query, with one prefix | `prefix_typed_time` for that prefix only |

In the test split, a repeated `session_id` means several queries in one session, not several keystrokes. **There is no timestamp per keystroke.** Only the average pace of a query can be derived: (end − start) ÷ number of prefixes.

On the first 200,000 rows of one train file (199,802 with positive duration):

| Measure | Value |
|---|---|
| prefixes per query | median 10 (middle half 6–14) |
| typing duration | median 6.3 s (middle half 4.2–9.9 s); maximum 85,924 s, so means are unusable |
| seconds per prefix | median 0.70 s (10th–90th percentile 0.33–2.0 s) |
| queries with a backspace | 38.9 % |
| queries ending before the final term is fully typed | 66 % *(consistent with picking a suggestion; not verified)* |

The pace table used by the simulator (3 files, durations trimmed to 1.73–48.61 s, 196,536 queries): 0.06 s at the 0th percentile, 0.33 at the 10th, 0.45 at the 25th, **0.67 at the median**, 1.05 at the 75th, 1.65 at the 90th.

## 1.2 Why the model is slower than the typist

Phase 1 latency (MPS, float32):

| Granularity | Mean (README) | Median | 90th percentile |
|---|---|---|---|
| next_word | 413 ms | 400 ms | 469 ms |
| partial_word | 456 ms | 430 ms | 585 ms |
| phrase | 1,098 ms | 1,050 ms | 1,322 ms |
| sentence | 3,181 ms | 3,005 ms | 5,508 ms |

A typist produces a key about every 0.67 s. Share of model calls whose answer arrives *before the next keystroke*, if the model is called on every keystroke:

| Typing style (modelled) | partial_word | next_word | phrase | sentence |
|---|---|---|---|---|
| steady | 68.7 % | 73.0 % | 21.8 % | 7.2 % |
| medium | 58.1 % | 61.7 % | 23.5 % | 8.4 % |
| bursty | 44.3 % | 46.7 % | 22.5 % | 10.9 % |

Phrase and sentence barely depend on the assumed style: calling on every keystroke wastes most phrase and sentence calls whatever the typing style. Word-level results swing by about 25 points with the style.

## 1.3 Pilot: seven rules, four numbers

Setup (`src/policy_eval.py`, seed 42, 10 simulated typists per example; 301 / 400 / 400 / 400 Qwen examples):

- Typing model: pace per typist from AmazonQAC (real); gaps inside the query from the Aalto gap shape (real, §2.1). The three modelled styles are kept as a robustness check (`results/policy_eval_style{0,1,2}.txt`).
- The rule decides at the cut point of each example: after the last typed character for partial_word, after the space following the prefix for the others.
- **Staleness is enforced in code.** While the model runs, the user keeps typing the true text (k characters). The suggestion is valid on arrival only if its first k characters match what was typed (`keystrokes_saved(prediction, reference) ≥ k`); otherwise it is worth 0.
- Value on arrival: characters still saved (next_word, partial_word, phrase) or semantic similarity (sentence), as chosen in `metric_choice.json`.
- **Missed** = share of the suggestions that always-invoke delivered which the rule did not deliver (never called, or arrived stale).

**Result with the real gap shape** (calls per 100 keystrokes / value per call / value kept / missed; full tables in `results/policy_eval_aalto.txt`)

| Rule | partial_word | next_word | phrase | sentence |
|---|---|---|---|---|
| always-invoke | 100 / 0.17 / 100 % / 0 % | 100 / 1.03 / 100 % / 0 % | 100 / 1.34 / 100 % / 0 % | 100 / 0.09 / 100 % / 0 % |
| every 5 keys | 18.7 / 0.14 / 18.9 % / 80.0 % | 17.8 / 0.93 / 14.5 % / 85.1 % | 17.8 / 1.27 / 15.1 % / 83.4 % | 17.8 / 0.09 / 15.9 % / 84.6 % |
| pause 0.3 s | 78.1 / 0.14 / 62.0 % / 37.7 % | 78.6 / 0.95 / 82.6 % / 14.3 % | 78.4 / 1.29 / 86.4 % / 13.1 % | 78.4 / 0.09 / 86.9 % / 13.6 % |
| pause 0.6 s | 46.8 / 0.14 / 35.1 % / 63.4 % | 48.1 / 0.95 / 58.8 % / 39.9 % | 47.2 / 1.33 / 62.7 % / 38.5 % | 47.3 / 0.09 / 59.0 % / 43.7 % |
| pause 1.0 s | 26.0 / 0.15 / 20.2 % / 77.1 % | 26.8 / 1.02 / 39.6 % / 60.7 % | 25.9 / 1.38 / 40.6 % / 60.1 % | 26.2 / 0.09 / 40.0 % / 62.3 % |
| word boundary | 18.1 / – / 0 % / 100 % | 20.1 / 1.03 / 100 % / 0 % | 20.1 / 1.34 / 100 % / 0 % | 20.1 / 0.09 / 100 % / 0 % |

Model-seconds per 100 keystrokes, always-invoke vs word boundary: partial_word 45.6 vs 8.3; next_word 41.3 vs 8.3; phrase 109.8 vs 22.1; sentence 318.1 vs 64.1. (Every-3-keys row omitted here; it misses 64–73 % everywhere.)

**What the pilot shows**

| Observation | Numbers |
|---|---|
| Fixed-interval rules miss most good suggestions | every 3 / 5 keys: missed 64–85 % in every cell |
| Waiting for a pause trades value for calls roughly one-for-one | pause 1.0 s: about a quarter of the calls, about 40 % of the value kept, 60 % missed; pause 0.3 s still needs 78 calls |
| Pause rules do not pay off at any granularity here | value per call within 0.93–1.03 of always-invoke at next_word; every pause variant either needs more than 50 % of the calls or misses more than 25 % |
| Word boundary delivers everything at about a fifth of the calls and cost | 20 calls and 0 % missed at next_word, phrase, sentence; cost drops by a factor of about 5 |
| Nothing works for partial_word | word boundary never fires mid-word (100 % missed); every other rule misses 38–80 % |

**Provisional selection under the proposed rule** (not yet confirmed by the supervisor): a rule qualifies if it uses at most 50 % of always-invoke's calls, misses at most 25 %, and has value per call at least as high as always-invoke's; among qualifying rules the one with the fewest calls wins.

| Granularity | Qualifying rule | Caveat |
|---|---|---|
| next_word, phrase, sentence | word boundary only | passes by construction, because the cut points sit at word ends (§4.2) |
| partial_word | none | needs a mid-word trigger rule that is not built yet |

**Real gap shape versus the invented styles.** At next_word, pause 1.0 s keeps 39.6 % of the value on 26.8 calls with the real shape; the modelled steady style gave 26.6 % and the medium style 47.1 %. Pause 0.3 s needs 78.6 calls with the real shape versus 72.9 with the medium style. The real shape lies between steady and medium, so the conclusion does not hinge on the invented bursty style.

**Reading rules for this table**

- "Calls per 100 keys" counts every keystroke typed before the cut point. "Value kept" is measured only at the cut points. Compare value *per call*, not the two percentages directly (see §4.3).
- A rule that never calls (word boundary at partial_word) scores 0 % kept; a rule that always lands on a word end scores 100 % by construction at the other three granularities (see §4.2).

---

# Part 2 — Approach

## 2.1 Real pace, real gap shape

`src/typing_sim.py` and `src/gap_shape.py`: each simulated typist draws an average pace from the real AmazonQAC distribution. Gaps inside the query are then drawn either from the **Aalto gap shape** (default result) or from an invented skewed distribution with an extra pause after each space (robustness check), and rescaled so the average gap equals the drawn pace.

**Aalto fit** (`fit_aalto_gaps.py`; [dataset page](https://userinterfaces.aalto.fi/136Mkeystrokes/), free for research and non-commercial use with attribution). The download is a folder of 168,595 per-participant files `N_keystrokes.txt` (tab-separated; columns `PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE`). 500 randomly chosen participants, seed 42. Per typed sentence, every gap between key presses is divided by that sentence's mean gap, so only the shape and the word-boundary effect are kept. SHIFT, CTRL, ALT, CAPS and META presses are dropped (not typed characters); sentences with fewer than 15 keystrokes or a gap over 10 s are skipped.

| Measure | Aalto (real) | Modelled steady / medium / bursty |
|---|---|---|
| sentences kept / skipped | 7,452 / 48 | – |
| gaps inside words / after a space | 287,190 / 53,395 | – |
| mean gap after a space ÷ mean gap inside words | **1.48×** | 1.0× / 2.0× / 3.0× |
| spread inside words (std of log gap) | **0.58** | 0.3 / 0.7 / 1.2 |
| gaps longer than 2× / 3× the typist's own average | 8.2 % / 2.7 % | – |

The invented medium and bursty styles overstated the pause after a space. The Aalto shape is stored as 1,001 quantiles per gap type in `aalto_gap_shape.csv`.

| Modelled style | `sigma` | `boundary_mult` |
|---|---|---|
| steady | 0.3 | 1.0 |
| medium | 0.7 | 2.0 |
| bursty | 1.2 | 3.0 |

## 2.2 The rules compared

In the order the research plan requires, simplest first:

| # | Rule | Implemented as |
|---|---|---|
| 1 | always-invoke (baseline) | call after every keystroke |
| 2 | fixed-interval | every 3 and every 5 keystrokes |
| 3 | pause-based | call only if no key within 0.3 / 0.6 / 1.0 s |
| 4 | boundary-based | call only right after a space |
| 5 | learned | not started |

## 2.3 The four numbers, always together

| Metric | In `policy_eval.py` |
|---|---|
| Cost | calls per 100 keystrokes **and** model-seconds per 100 keystrokes (calls × mean latency) |
| Quality | value per call, per `metric_choice.json` |
| Invocation rate | calls per 100 keystrokes |
| Missed opportunities | share of always-invoke's delivered suggestions that the rule did not deliver |

---

# Part 3 — Implementation Guideline

## 3.1 Dependency order

```
  A0  AmazonQAC check ──► A1  pace table ──► A2  typing_sim ──► A3  timing checks
                                                  │
                                                  └──► A4  policy_eval (pilot)
                                                          │
        A5  Aalto gap model ─────────────────────────────►┤  replaces modelled gaps
        A6  >200 messages, message_id, split by message ──┤
                                                          └──► A7  learned ──► A8  AESLC
```

## 3.2 Stage notes

### A1 — `make_pace_table.py`
Reads 70,000 rows from each of 3 train files (every 7th of 20), parses mixed timestamp formats (`format="mixed"`; some rows have no milliseconds), drops queries with fewer than 2 prefixes, trims durations to the 1st–99th percentile, and saves 1,001 quantile points. It contains a machine-specific cache path.

### A2 — `src/typing_sim.py`
`simulate_typing(text, pace, sigma, boundary_mult, rng)` returns `(time, prefix_typed_so_far)` per keystroke. Everything is seeded.

### A4 — `src/policy_eval.py`
Read-only on `phase1_results_v2.csv`. Run from the repo root: `uv run python -m src.policy_eval [0|1|2|aalto]` (0/1/2 = modelled styles, `aalto` = real gap shape).

> ⚠ `GOOD_SIM = 0.4` is the only threshold in the script. It defines a "good" similarity-scored suggestion for the sentence granularity and is provisional.

### A5 — Aalto gap model
`uv run python fit_aalto_gaps.py <folder or zip> [participants]` reads the participant files, drops modifier keys, normalises gaps per sentence, splits them into gaps after a space and all others, and writes `aalto_gap_shape.csv`. `simulate_typing_empirical` in `src/gap_shape.py` samples from it by inverse CDF and rescales to the AmazonQAC pace. The default of 3,000 participants was not needed; 500 is enough for a stable shape and can be raised.

### A6 — Data for the learned rule and the headline result
Phase 1 has about 80 messages (400 examples ÷ 5 cut points); the research plan requires more than 200. `N_MESSAGES_FULL = 260` exists, but a run overwrites results. Before running: commit, write to a new file, and add `message_id` and cut-point columns so all 5 cut points of a message land on the same side of the train/test split.

## 3.3 Verification

| # | Check | Expected | Status |
|---|---|---|---|
| 1 | keystroke rate with zero delay | 9.9 / 25.9 / 7.1 / 3.9 % | ✅ matches exactly |
| 2 | metric read from `metric_choice.json` | no "could not read" warning | ✅ none in the Aalto run |
| 3 | seeded | same output twice | by construction (seed 42) |
| 4 | all typing models run | tables for styles 0, 1, 2 and aalto in `results/` | ✅ |
| 5 | `pytest tests/` | green incl. new tests | tests not written yet |

## 3.4 Reproduce

```bash
uv run python make_pace_table.py             # needs the local AmazonQAC cache (~60 GB)
uv run python src/typing_sim.py              # gap statistics per typing style
uv run python src/latency_vs_gaps.py         # share of answers arriving before the next key
uv run python fit_aalto_gaps.py ~/Downloads/Keystrokes/files 500   # needs the Aalto download
uv run python -m src.policy_eval aalto       # real gap shape
uv run python -m src.policy_eval 0           # steady typist (robustness)
uv run python -m src.policy_eval 1           # medium
uv run python -m src.policy_eval 2           # bursty
```

---

# Part 4 — Open Issues

## 4.1 Gap shape: real, but from copy-typing
The pause-based conclusions now rest on measured gap shape rather than invented styles, and the real shape lies between the steady and medium styles. See §4.7 for the remaining limit.

## 4.2 Cut points sit at fixed positions
next_word, phrase and sentence examples are cut at word ends, and partial_word examples mid-word. A boundary rule therefore cannot miss any of the first three on this data, and cannot serve partial_word at all. A combined rule (boundary for the first three, a separate mid-word trigger for partial_word) is not tested yet.

## 4.3 Invocation rate and value are measured at different positions
Calls per 100 keystrokes averages over every keystroke before the cut; value is measured only at the cut. In styles with pauses after spaces, cut points have longer gaps than the average keystroke, which flatters pause rules in "value kept". Value per call is the safer comparison.

## 4.4 Staleness for the similarity-scored granularity
For sentence, a suggestion counts only if its first typed characters match literally, which is strict for paraphrases. The rule needs a decision from the supervisor.

## 4.5 Latency numbers are not production latency
MPS, float32, no prefix cache. Phase 3 (GGUF, with and without caching) will change both the timing and possibly the winner. Report every latency with its runtime.

## 4.6 AmazonQAC pace includes time after the last keystroke
Duration runs until `search_time`, so per-keystroke pace is slightly overestimated.

## 4.7 Aalto is probably copy-typing
As far as I can tell, participants type sentences shown to them, so thinking pauses are rarer than when composing (only 8.2 % of gaps exceed twice the typist's own average). This is why overall pace stays with AmazonQAC. The dataset's `readme.txt` has not been read yet; the claim is unverified. Real pauses while composing would make pause rules look better than here, so the pause-rule result is the one to re-check on other data.

## 4.8 Sample size
The pilot uses about 80 messages; the research plan requires more than 200.

---

## Files

| File | What it is |
|---|---|
| `make_pace_table.py` | builds the pace table from the local AmazonQAC cache |
| `amazonqac_pace_quantiles.csv` | 1,001 quantile points of seconds per prefix (small, committed) |
| `src/typing_sim.py` | typing simulator |
| `src/latency_vs_gaps.py` | share of answers arriving before the next key |
| `src/policy_timing.py` | timing-only comparison of rules |
| `src/policy_eval.py` | the pilot: seven rules, four numbers |
| `fit_aalto_gaps.py` | fits the real gap shape from the Aalto files |
| `aalto_gap_shape.csv` | 1,001 quantiles of relative gaps (after a space / other), small, committed |
| `src/gap_shape.py` | samples gaps from that table |
| `inspect_aalto.py` | inspects an Aalto zip without unzipping (not needed for the folder download) |
| `results/` | saved outputs of `policy_eval` for aalto and styles 0, 1, 2 |
| `phase1_results_v2.csv` | frozen Phase 1 results, read-only |