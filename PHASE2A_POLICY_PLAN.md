# Phase 2a — When to Invoke the Model

**One-line summary:** Phase 2a compares rules for *when* to call the model while a user types: 240 OASST2 messages, real typing pace (AmazonQAC), a real gap shape (Aalto), the single-suggestion latency, and all four numbers the research plan requires. A simple rule already cuts cost a lot, but none meets the proposed targets for mid-word suggestions, which is the condition for building the learned policy (Stage A7). Four decisions need feedback (next section) before Stage A7 and the AESLC run start.

---

## Progress

- [x] **Stage A0** — AmazonQAC timestamp check *(answer: timestamps are per query, not per prefix or keystroke — see §1.1)*
- [x] **Stage A1** — Pace table from AmazonQAC: `make_pace_table.py` → `amazonqac_pace_quantiles.csv`
      *(210,000 train rows read from 3 of 20 files, 196,536 kept after trimming; median 0.67 s per typed prefix)*
- [x] **Stage A2** — `src/typing_sim.py` *(real pace per typist; gaps either modelled or, since A5, from the real Aalto shape, see §2.1)*
- [x] **Stage A3** — Timing-only checks: `src/latency_vs_gaps.py`, `src/policy_timing.py`
- [x] **Stage A4** — Pilot: `src/policy_eval.py` *(7 rules, four numbers, staleness enforced in code)*
      *(gate passed: keystroke rate with zero delay reproduces the metric plan's 9.9 / 25.9 / 7.1 / 3.9 % exactly)*
- [x] **Stage A5** — Real gap shape from the Aalto 136M Keystrokes dataset: `fit_aalto_gaps.py` → `aalto_gap_shape.csv`, `src/gap_shape.py`, `policy_eval aalto`
      *(500 of 168,595 participant files, 7,452 sentences kept; word-boundary effect 1.48×, spread 0.58 — milder than the modelled styles, see §2.1)*
- [x] **Stage A6** — More than 200 messages with `message_id` saved: `src/run_eval_v3.py` → `phase1_results_v3.csv`; evaluated by `src/policy_eval_v3.py`
      *(240 evaluation messages, 4,510 Qwen examples, 18,040 rows; all 382 next_word prefixes of the v2 run reappear in v3; intervals are computed over messages)*
- [ ] **Stage A7** — Learned policy *(a gap is left by every simple rule, see §1.3)* ← **YOU ARE HERE**
- [ ] **Stage A8** — One run on AESLC
- [ ] `tests/` for `typing_sim` and `policy_eval`

> ⚠ **§1.3 is a development result.** It rests on a simulated typist, fixed cut positions (§4.2), copy-typing gap shapes (§4.7) and a selection rule that is still a proposal. A check on a second dataset (Stage A8) is still open.

---

## Feedback required

Stages A0 to A6 are done. The four points below need feedback on what should be done; until then the work continues with the default in the third column. Stage A7 and the AESLC run wait for answers 1 and 3.

| # | Question | Default until feedback arrives | Details |
|---|---|---|---|
| 1 | **Selection rule.** Are "at most 50 % of always-invoke's calls, at most 25 % missed, value per call at least 0.95× always-invoke's" the right targets, or should only the trade-off curve (calls against missed) be reported, with no single winner? | the curve is reported; the thresholds are marked as a proposal | §1.3, §4.9 |
| 2 | **Latency definition.** Should cost use the time to the first suggestion (what a user waits for: 0.11 / 0.10 / 0.29 / 0.72 s) or the five-candidate time that Phase 1 reported (0.54 / 0.51 / 1.51 / 3.57 s)? | first suggestion; both columns are kept in `phase1_results_v3.csv` | §1.2, §4.10 |
| 3 | **Cut points and weights.** The Phase 1 cut points sit at word starts (next_word, phrase, sentence) or inside a word (partial_word). Both are pooled with weights 3.9 : 1 (letters per word, taken from the data). Is that a fair test, or should every keystroke position of a message be evaluated? | the pooled word-level mix stays | §1.3, §4.2 |
| 4 | **Staleness for sentence.** A suggestion counts only if its first typed characters match literally, which is strict for paraphrases. Should an embedding-based rule be used instead? | literal match | §4.4 |

Open for comments, no decision needed: the Aalto data is copy-typing (§4.7), and the plan for the learned policy (§3.2, A7). Documentation updates (the README latency table) follow once question 2 is settled.

---

# Part 1 — The Problem and What the Results Show

## 1.1 What AmazonQAC can and cannot tell us

The research plan says not to invent pause timings and to confirm whether AmazonQAC timestamps are per keystroke or per prefix. The downloaded files were checked directly:

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

## 1.2 Latency: what the model costs per suggestion

`run_eval.py` times five sampled candidates in a row (for the top-5 list), but a user is shown one suggestion. Stage A6 therefore records both:

| Granularity | Time to first suggestion (used) | Time for 5 candidates (Phase 1, README) |
|---|---|---|
| partial_word | 0.11 s | 0.54 s |
| next_word | 0.10 s | 0.51 s |
| phrase | 0.29 s | 1.51 s |
| sentence | 0.72 s | 3.57 s |

*(means over the v3 run, MPS float32; the Phase 1 README table of 413 / 456 / 1,098 / 3,181 ms is the five-candidate time of the 80-message run.)* The pilot used the five-candidate time as arrival time, which overstated both cost and staleness by roughly a factor of five. A typist produces a key about every 0.67 s, so a word-level suggestion now arrives before the next key; a sentence takes about one key gap. Staleness is therefore a small effect here: with the corrected latency, missed suggestions are almost entirely suggestions the rule never asked for (see §1.3). Phase 3 (GGUF, prefix cache) will change these times again. *Feedback required (question 2).*

## 1.3 Results: fourteen rules, four numbers, 240 messages

Setup (`src/policy_eval_v3.py aalto`, seed 42, 10 simulated typists per example; 910 / 1,200 / 1,200 / 1,200 Qwen examples from 240 messages):

- Typing model: pace per typist from AmazonQAC (real); gaps from the Aalto gap shape (real, §2.1). The three modelled styles remain available as a robustness check (`policy_eval_v3.py 0|1|2`).
- The rule decides at the cut point of each example: after the last typed character for partial_word (a mid-word position), after the space following the prefix for the others (a word-start position).
- **Staleness is enforced in code.** While the model runs, the user keeps typing the true text (k characters). The suggestion is valid on arrival only if its first k characters match what was typed (`keystrokes_saved(prediction, reference) ≥ k`); otherwise it is worth 0.
- Value on arrival: characters still saved (next_word, partial_word, phrase) or semantic similarity (sentence), as chosen in `metric_choice.json`.
- **Missed** = share of the suggestions that always-invoke delivered which the rule did not deliver (never called, or arrived stale). The 95 % interval is a bootstrap over messages (1,000 resamples), because the cut points of one message are not independent.
- **Word-level mix:** a real message has about 3.9 mid-word keystrokes for every space. The mix pools partial_word and next_word examples with weights 3.9 : 1 to evaluate a rule that has to serve both kinds of position.

**Missed suggestions per granularity** (full tables with cost, value per call and intervals in `results/policy_eval_v3_aalto.txt`)

| Rule | calls / 100 keys | partial_word | next_word | phrase | sentence |
|---|---|---|---|---|---|
| always-invoke | 100 | 0 % | 0 % | 0 % | 0 % |
| every 3 keys | 31.5 | 74.6 % | 68.1 % | 67.5 % | 68.3 % |
| every 5 keys | 17.8 | 83.5 % | 80.5 % | 81.5 % | 79.4 % |
| pause 0.2 s | 90.6 | 23.0 % | 4.9 % | 5.6 % | 9.0 % |
| pause 0.3 s | 78.4 | 36.3 % | 12.8 % | 13.0 % | 16.8 % |
| pause 0.4 s | 66.5 | 48.3 % | 21.7 % | 21.0 % | 25.7 % |
| pause 0.6 s | 47.5 | 63.7 % | 38.4 % | 37.4 % | 41.2 % |
| pause 0.8 s | 34.8 | 73.2 % | 52.4 % | 51.1 % | 53.7 % |
| pause 1.0 s | 26.3 | 80.0 % | 62.3 % | 60.9 % | 63.2 % |
| pause 1.2 s | 20.3 | 84.8 % | 70.4 % | 69.0 % | 70.3 % |
| word boundary | 20.0 | 100 % | 0 % | 0 % | 0 % |
| boundary + mid-word 0.4 s | 70.4 | 48.3 % | 0 % | 0 % | 0 % |
| boundary + mid-word 0.6 s | 54.8 | 63.7 % | 0 % | 0 % | 0 % |
| boundary + mid-word 1.0 s | 38.4 | 80.0 % | 0 % | 0 % | 0 % |

*(calls per 100 keystrokes from the next_word table; the other granularities differ by at most 0.4. "boundary + mid-word T" fires after every space and, inside a word, only after a pause longer than T.)*

**Word-level mix** (partial_word + next_word, keystrokes saved)

| Rule | calls / 100 keys | model-s / 100 keys | value per call | value kept | missed (95 % interval) |
|---|---|---|---|---|---|
| always-invoke | 100 | 11.2 | 0.40 | 100 % | 0 % |
| every 3 keys | 31.9 | 3.6 | 0.36 | 29.8 % | 72.1 % (67–77) |
| every 5 keys | 18.2 | 2.0 | 0.33 | 17.3 % | 82.4 % (78–86) |
| pause 0.2 s | 90.7 | 10.2 | 0.38 | 87.5 % | 16.1 % (15–18) |
| pause 0.3 s | 78.5 | 8.8 | 0.39 | 77.0 % | 27.4 % (26–29) |
| pause 0.4 s | 66.4 | 7.5 | 0.39 | 66.4 % | 38.2 % (36–40) |
| pause 0.6 s | 47.4 | 5.3 | 0.41 | 50.0 % | 54.1 % (52–56) |
| pause 0.8 s | 34.8 | 3.9 | 0.42 | 38.2 % | 65.3 % (63–67) |
| pause 1.0 s | 26.2 | 2.9 | 0.43 | 29.5 % | 73.3 % (72–75) |
| pause 1.2 s | 20.3 | 2.3 | 0.44 | 23.1 % | 79.4 % (78–81) |
| word boundary | 18.4 | 2.1 | 0.91 | 57.5 % | 62.2 % (57–66) |
| boundary + mid-word 0.4 s | 70.0 | 7.8 | 0.44 | 79.7 % | 30.0 % (27–33) |
| boundary + mid-word 0.6 s | 54.1 | 6.1 | 0.50 | 72.8 % | 39.6 % (36–43) |
| boundary + mid-word 1.0 s | 37.4 | 4.2 | 0.62 | 65.8 % | 49.8 % (46–53) |

**What the results show**

| Observation | Numbers |
|---|---|
| Fixed-interval rules miss most good suggestions | missed 67–84 % at every granularity |
| Pause rules trade value for calls roughly one-for-one | pause 1.0 s: about a quarter of the calls, 61–80 % missed |
| At word-start positions a boundary rule delivers everything at a fifth of the calls | 20 calls per 100 keys, 0 % missed, same value per call; model time falls from 10.4 to 2.1 (next_word) and from 72.0 to 14.4 (sentence) model-seconds per 100 keys. This holds by construction (§4.2) |
| Mid-word positions are the hard case | no rule at partial_word has fewer than 50 % of the calls and fewer than 25 % missed; pause 0.2 s misses 23 % but still needs 91 % of the calls |
| The combined rule beats plain pause rules at the same cost | at about 37 % of the calls it misses 49.8 %, pause 0.8 s at 35 % misses 65.3 %; at about 54 % of the calls 39.6 % against 54.1 % for pause 0.6 s at 47 % |
| Word boundary alone beats every pause rule below about 25 % of the calls | 18.4 calls, 62.2 % missed, against 79.4 % for pause 1.2 s at 20.3 calls |
| Stale answers are a small part of the loss | with the single-suggestion latency, next_word pause 0.6 s misses 38.4 % (pilot with five-candidate latency: 39.9 %) |

**Proposed selection rule** (*feedback required, question 1*): a rule qualifies if it uses at most 50 % of always-invoke's calls, misses at most 25 %, and keeps value per call at 0.95 × always-invoke's or more; the qualifying rule with the fewest calls wins. `policy_eval_v3.py` prints a PASS column for it.

| Position type | Qualifying rule | Caveat |
|---|---|---|
| word start (next_word, phrase, sentence) | word boundary (and the combined rule) | passes by construction, because the Phase 1 cut points sit at word starts (§4.2) |
| mid-word (partial_word) | none | the best simple rules miss 48–64 % at 47–70 % of the calls |
| word-level mix | none | the closest are the combined rule at 0.4 s (70 calls, 30 % missed) and 0.6 s (54 calls, 40 % missed) |

**Reading rules for these tables**

- "Calls per 100 keys" counts every keystroke typed before the cut point. "Value kept" is measured only at the cut points. Compare value *per call*, not the two percentages directly (see §4.3).
- The 25 % and 50 % thresholds are a proposal. The research plan's exit criterion is that a policy lowers the invocation rate versus always-invoke while keeping usefulness above the hand-validated threshold, with missed opportunities and staleness reported; the trade-off curve (calls against missed) can be reported without a single winner.

---

# Part 2 — Approach

## 2.1 Real pace, real gap shape

In `src/typing_sim.py` and `src/gap_shape.py`, the average pace of each simulated typist is drawn from the real AmazonQAC distribution. The gaps inside the query are then drawn either from the **Aalto gap shape** (main result) or from an invented skewed distribution with an extra pause after each space (robustness check), and rescaled so the average gap equals the drawn pace.

**Aalto fit** (`fit_aalto_gaps.py`; [dataset page](https://userinterfaces.aalto.fi/136Mkeystrokes/), free for research and non-commercial use with attribution). The download is a folder of 168,595 per-participant files `N_keystrokes.txt` (tab-separated; columns `PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE`). 500 randomly chosen participants were used (seed 42). For each typed sentence, every gap between key presses is divided by that sentence's mean gap, so only the shape and the word-boundary effect are kept. SHIFT, CTRL, ALT, CAPS and META presses are dropped (not typed characters); sentences with fewer than 15 keystrokes or a gap over 10 s are skipped.

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
| 3 | pause-based | call only if no key within 0.2 / 0.3 / 0.4 / 0.6 / 0.8 / 1.0 / 1.2 s |
| 4 | boundary-based | call only right after a space |
| 4b | boundary + mid-word pause | call right after every space, and inside a word only after a pause of 0.4 / 0.6 / 1.0 s |
| 5 | learned | Stage A7, not started |

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
70,000 rows are read from each of 3 train files (every 7th of 20); mixed timestamp formats are parsed (`format="mixed"`; some rows have no milliseconds), queries with fewer than 2 prefixes are dropped, durations are trimmed to the 1st–99th percentile, and 1,001 quantile points are saved. The script contains a machine-specific cache path that has to be changed before anyone else can run it.

### A2 — `src/typing_sim.py`
`simulate_typing(text, pace, sigma, boundary_mult, rng)` returns `(time, prefix_typed_so_far)` per keystroke. Everything is seeded.

### A4 — `src/policy_eval.py`
Read-only on `phase1_results_v2.csv`. Run from the repo root: `uv run python -m src.policy_eval [0|1|2|aalto]` (0/1/2 = modelled styles, `aalto` = real gap shape).

> ⚠ `GOOD_SIM = 0.4` is the only threshold in the script. It defines a "good" similarity-scored suggestion for the sentence granularity and is provisional.

### A5 — Aalto gap model
`uv run python fit_aalto_gaps.py <folder or zip> [participants]` reads the participant files, drops modifier keys, normalises gaps per sentence, splits them into gaps after a space and all others, and writes `aalto_gap_shape.csv`. `simulate_typing_empirical` in `src/gap_shape.py` samples from it by inverse CDF and rescales to the AmazonQAC pace. 500 participants were used, which gave a stable shape; the number can be raised.

### A6 — Data for the learned rule and the headline result
`src/run_eval_v3.py` loads 260 messages (the same shuffle as the v2 run); the first 20 fit the baselines, the other 240 are evaluated. Every row carries `message_id`, `cut_index`, `prefix_words` and `msg_words`. `latency_top1_ms` is the time until the first suggestion exists, `latency_ms` the time for all five candidates. Every example is seeded. Results are appended per message to a `.jsonl` file, so a stopped run resumes, and the `.csv` is written at the end (about 36 s per message, about 2 hours in total). A one-line warm-up call keeps the first row free of start-up latency. `phase1_results_v2.csv` is never touched; the hand labels still belong to it.

`src/policy_eval_v3.py` runs the evaluation (§1.3); it writes `results/policy_eval_v3_summary.csv` (one row per group and rule) for the trade-off chart.

### A7 — Learned policy
The goal is to beat the combined rule at the same number of calls. Training uses features known *before* the call (letters typed in the current word, position in the message, pause length, previous word) to predict whether a call delivers value (`ks0 > 0` and not stale). Train and test are split by `message_id` (all cut points of a message on the same side), and the comparison uses the same trade-off curve (calls against missed). Only about 900 mid-word and 1,200 word-start examples exist, so a simple model (logistic regression or small gradient boosting) fits. It starts once questions 1 and 3 are answered.

## 3.3 Verification

| # | Check | Expected | Status |
|---|---|---|---|
| 1 | keystroke rate with zero delay (v2 file) | 9.9 / 25.9 / 7.1 / 3.9 % | ✅ matches exactly |
| 2 | metric read from `metric_choice.json` | no "could not read" warning | ✅ none in the v3 run |
| 3 | seeded | same output twice | by construction (seed 42; every generation seeded) |
| 4 | v3 contains the v2 messages | all v2 next_word prefixes reappear | ✅ 382 of 382 |
| 5 | pilot re-run independently by a teammate | same rows | ✅ every-3, every-5 and pause 0.6 s rows identical on the v2 file |
| 6 | all typing models run | tables for aalto and styles 0, 1, 2 | aalto ✅ (v2 file: all four) |
| 7 | `pytest tests/` | green incl. new tests | tests not written yet |

## 3.4 Reproduce

```bash
uv run python make_pace_table.py             # needs the local AmazonQAC cache (~60 GB)
uv run python fit_aalto_gaps.py ~/Downloads/Keystrokes/files 500   # needs the Aalto download
uv run python -m src.run_eval_v3             # 240 messages, about 2.4 h; resumes if stopped
uv run python -m src.policy_eval_v3 aalto    # the results of section 1.3
uv run python -m src.policy_eval_v3 1        # robustness: medium modelled typist
uv run python -m src.policy_eval 1           # earlier 80-message pilot on phase1_results_v2.csv
```

---

# Part 4 — Open Issues

## 4.1 Gap shape: real, but from copy-typing
The pause-based conclusions rest on the measured gap shape, which lies between the steady and medium modelled styles. See §4.7 for the remaining limit.

## 4.2 Cut points sit at fixed positions
The next_word, phrase and sentence examples are cut at word starts; the partial_word examples inside a word. A boundary rule therefore cannot miss a word-start example on this data, and cannot serve a mid-word one. The word-level mix (§1.3) evaluates both position types with the weights of real typing (about 3.9 : 1, estimated from word lengths, not measured). It looks like the fair test for a rule that must cover both, and random cut positions in the generation step would add little, because both position types already exist. **Feedback required (question 3).**

## 4.3 Invocation rate and value are measured at different positions
Calls per 100 keystrokes averages over every keystroke before the cut; value is measured only at the cut. In styles with pauses after spaces, cut points have longer gaps than the average keystroke, which flatters pause rules in "value kept". Value per call is the safer comparison.

## 4.4 Staleness for the similarity-scored granularity
For sentence, a suggestion counts only if its first typed characters match literally, which is strict for paraphrases. With the corrected latency the effect is small, but the rule still needs a decision. **Feedback required (question 4).**

## 4.5 Latency numbers are not production latency
MPS, float32, no prefix cache. Phase 3 (GGUF, with and without caching) will change both the timing and possibly the winner. Every latency is reported with its runtime and with whether it is the first suggestion or five candidates.

## 4.6 AmazonQAC pace includes time after the last keystroke
Duration runs until `search_time`, so per-keystroke pace is slightly overestimated.

## 4.7 Aalto is probably copy-typing
As far as can be told, participants type sentences shown to them, so thinking pauses are rarer than when composing (only 8.2 % of gaps exceed twice the typist's own average). This is why the overall pace comes from AmazonQAC. The dataset's `readme.txt` has not been read yet, so the claim is unverified. Real pauses while composing would make pause rules look better than here, so the pause-rule result is the one to re-check on other data.

## 4.8 Sample size
The v3 run has 240 evaluation messages (the research plan requires more than 200). The intervals are over messages and do not include the randomness of the simulated typists.

## 4.9 The selection rule is a proposal
The 50 % / 25 % / 0.95 thresholds are not from the research plan. Either the trade-off curve (calls against missed) is reported with the choice left open, or the thresholds are confirmed before the learned policy is compared. **Feedback required (question 1).**

## 4.10 Phase 1 latency in the README
The README latency table (413 / 456 / 1,098 / 3,181 ms) is the five-candidate time; the measured first-suggestion time is 0.11 / 0.10 / 0.29 / 0.72 s. The README is updated once question 2 is settled. **Feedback required (question 2).**

## Files

| File | What it is |
|---|---|
| `make_pace_table.py` | builds the pace table from the local AmazonQAC cache |
| `amazonqac_pace_quantiles.csv` | 1,001 quantile points of seconds per prefix (small, committed) |
| `fit_aalto_gaps.py` | fits the real gap shape from the Aalto files |
| `aalto_gap_shape.csv` | 1,001 quantiles of relative gaps (after a space / other), small, committed |
| `src/gap_shape.py` | samples gaps from that table |
| `src/typing_sim.py` | typing simulator |
| `src/latency_vs_gaps.py` | share of answers arriving before the next key (five-candidate latency, pilot) |
| `src/policy_timing.py` | timing-only comparison of rules (pilot) |
| `src/policy_eval.py` | the earlier pilot on `phase1_results_v2.csv` (five-candidate latency) |
| `src/run_eval_v3.py` | Phase 1 re-run at scale: 240 messages, `message_id`, first-suggestion latency, seeded, resumable |
| `src/policy_eval_v3.py` | the results of §1.3: rules, intervals over messages, word-level mix, PASS column |
| `phase1_results_v3.csv` | results of the v3 run (the raw `.jsonl` is not committed) |
| `results/` | saved outputs: pilot (aalto, styles 0/1/2) and `policy_eval_v3_aalto.txt`, `policy_eval_v3_summary.csv` |
| `inspect_aalto.py` | inspects an Aalto zip without unzipping (not needed for the folder download) |
| `phase1_results_v2.csv` | frozen Phase 1 results, read-only; hand labels belong to it |