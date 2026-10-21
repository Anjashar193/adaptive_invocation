# Phase 2a — When to Invoke the Model

**One-line summary:** Phase 2a compares rules for *when* to call the model while a user types. It uses 240 OASST2 messages, real typing pace (AmazonQAC), a real gap shape (Aalto), the single-suggestion latency, and the four numbers the research plan requires. Pauses do not help; how much has been typed does. Only sentence suggestions meet the proposed targets (about 40 % of the calls and about 23 % missed, by calling after about 11 words). A learned policy did not beat these simple rules. Four decisions need feedback (next section) before the full-session run and the AESLC run start.

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
- [x] **Stage A7** — Learned policy and position rules: `src/learned_policy.py`
      *(variant chosen on 120 dev messages and scored on 120 test messages; pauses do not help, letters and words typed do; the learned policy does not beat the best simple position rule, see §1.4)*
- [ ] **Stage A7b** — Ceiling analysis: the best any policy could do with the current inputs ← **YOU ARE HERE**
- [ ] **Stage A7c** — Full typing sessions: suggestions at every keystroke for about 30 messages
- [ ] **Stage A8** — One run on AESLC
- [ ] `tests/` for `typing_sim` and `policy_eval`

> ⚠ **The results are development results.** They rest on a simulated typist, 5 fixed cut points per message (§4.2), copy-typing gap shapes (§4.7), and a selection rule that is still a proposal. The test half of the messages has been looked at several times, so a fresh set of messages and a check on a second dataset (Stage A8) are still open.

---

## Feedback required

Stages A0 to A7 are done. The four points below need feedback on what should be done; until then the work continues with the default in the third column. The full-session run (A7c) and the AESLC run wait for answers 1 and 3.

| # | Question | Default until feedback arrives | Details |
|---|---|---|---|
| 1 | **Selection rule.** Are "at most 50 % of always-invoke's calls, at most 25 % missed, value per call at least 0.95× always-invoke's" the right targets, or should only the trade-off curve (calls against missed) be reported, with no single winner? Is a simple threshold on words and letters typed an acceptable answer to "when to invoke"? | the curve is reported; the thresholds are marked as a proposal | §1.3, §1.4, §4.9 |
| 2 | **Latency definition.** Should cost use the time to the first suggestion (what a user waits for: 0.11 / 0.10 / 0.29 / 0.72 s) or the five-candidate time that Phase 1 reported (0.54 / 0.51 / 1.51 / 3.57 s)? | first suggestion; both columns are kept in `phase1_results_v3.csv` | §1.2, §4.10 |
| 3 | **Cut points and weights.** The Phase 1 cut points sit at word starts (next_word, phrase, sentence) or inside a word (partial_word), 5 per message. Both are pooled with weights 3.9 : 1 (letters per word, taken from the data). Should every keystroke position of a message be evaluated instead (Stage A7c)? | the pooled word-level mix stays; A7c is planned | §1.3, §4.2 |
| 4 | **Staleness for sentence.** A suggestion counts only if its first typed characters match literally, which is strict for paraphrases. Should an embedding-based rule be used instead? | literal match | §4.4 |

Open for comments, no decision needed: the Aalto data is copy-typing (§4.7), and the plan for A7b and A7c (§3.2). Documentation updates (the README latency table) follow once question 2 is settled.

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

*(means over the v3 run, MPS float32; the Phase 1 README table of 413 / 456 / 1,098 / 3,181 ms is the five-candidate time of the 80-message run.)* The pilot used the five-candidate time as arrival time, which overstated both cost and staleness by roughly a factor of five. A typist produces a key about every 0.67 s, so a word-level suggestion now arrives before the next key, and a sentence takes about one key gap. Staleness is therefore a small effect: with the corrected latency, missed suggestions are almost entirely suggestions the rule never asked for. Phase 3 (GGUF, prefix cache) will change these times again. *Feedback required (question 2).*

## 1.3 Results for timing rules: fourteen rules, four numbers, 240 messages

Setup (`src/policy_eval_v3.py aalto`, seed 42, 10 simulated typists per example; 910 / 1,200 / 1,200 / 1,200 Qwen examples from 240 messages):

- **Typing model:** pace per typist from AmazonQAC (real); gaps from the Aalto gap shape (real, §2.1). The three modelled styles remain available as a robustness check (`policy_eval_v3.py 0|1|2`).
- **Decision point:** the rule decides at the cut point of each example. That is after the last typed character for partial_word (a mid-word position), and after the space following the prefix for the others (a word-start position).
- **Staleness is enforced in code.** While the model runs, the user keeps typing the true text (k characters). The suggestion is valid on arrival only if its first k characters match what was typed (`keystrokes_saved(prediction, reference) ≥ k`); otherwise it is worth 0.
- **Value on arrival:** characters still saved (next_word, partial_word, phrase) or semantic similarity (sentence), as chosen in `metric_choice.json`.
- **Missed** = the share of the suggestions that always-invoke delivered which the rule did not deliver (never called, or arrived stale). The 95 % interval is a bootstrap over messages (1,000 resamples), because the cut points of one message are not independent.
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

*(Calls per 100 keystrokes come from the next_word table; the other granularities differ by at most 0.4. "boundary + mid-word T" fires after every space and, inside a word, only after a pause longer than T.)*

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

**What the timing rules show**

| Observation | Numbers |
|---|---|
| Fixed-interval rules miss most good suggestions | missed 67–84 % at every granularity |
| Pause rules trade value for calls roughly one-for-one | pause 1.0 s: about a quarter of the calls, 61–80 % missed |
| At word-start positions a boundary rule delivers everything at a fifth of the calls | 20 calls per 100 keys, 0 % missed, same value per call; model time falls from 10.4 to 2.1 (next_word) and from 72.0 to 14.4 (sentence) model-seconds per 100 keys. This holds by construction (§4.2) |
| Mid-word positions are the hard case for timing rules | no timing rule at partial_word has fewer than 50 % of the calls and fewer than 25 % missed; pause 0.2 s misses 23 % but still needs 91 % of the calls |
| The combined rule beats plain pause rules at the same cost | at about 37 % of the calls it misses 49.8 %, pause 0.8 s at 35 % misses 65.3 %; at about 54 % of the calls 39.6 % against 54.1 % for pause 0.6 s at 47 % |
| Stale answers are a small part of the loss | with the single-suggestion latency, next_word pause 0.6 s misses 38.4 % (pilot with five-candidate latency: 39.9 %) |

**Proposed selection rule** (*feedback required, question 1*): a rule qualifies if it uses at most 50 % of always-invoke's calls, misses at most 25 %, and keeps value per call at 0.95 × always-invoke's or more. The qualifying rule with the fewest calls wins. The thresholds are a proposal.

| Position type | Qualifying timing rule | Caveat |
|---|---|---|
| word start (next_word, phrase, sentence) | word boundary (and the combined rule) | passes by construction, because the Phase 1 cut points sit at word starts (§4.2) |
| mid-word (partial_word) | none | the best timing rules miss 48–64 % at 47–70 % of the calls |
| word-level mix | none | the closest are the combined rule at 0.4 s (70 calls, 30 % missed) and 0.6 s (54 calls, 40 % missed) |

**Reading rules for these tables**

- "Calls per 100 keys" counts every keystroke typed before the cut point. "Value kept" is measured only at the cut points. Compare value *per call*, not the two percentages directly (§4.3).
- The 25 % and 50 % thresholds are a proposal. The research plan's exit criterion is that a policy lowers the invocation rate versus always-invoke while keeping usefulness above the hand-validated threshold, with missed opportunities and staleness reported; the trade-off curve (calls against missed) can be reported without a single winner.

## 1.4 Results for position rules and the learned policy

The timing rules above leave a gap, so Stage A7 tested two more things: simple rules that look at how much has been typed, and a learned policy.

**Setup** (`src/learned_policy.py`; the same simulation, staleness rule and latency as §1.3):

- The 240 messages are split once at random into a **dev half** and a **test half** (120 each). The learned variant is chosen on the dev half by 5-fold cross-validation split by message. It is then trained on the whole dev half and scored on the test half, which is never used for a choice. All numbers below are test-half numbers.
- **Position rules** (not learned): call once the current word has n letters; call at every word start and mid-word once the word has n letters; call once m words are typed; call at word starts once m words are typed.
- **Learned policy:** logistic regression and small gradient-boosted trees. Inputs are known before the call: pause lengths and typing speed, letters in the current word, words typed, prefix length. The label is whether a call delivers a suggestion that is still valid on arrival. Variants with and without a short wait before a mid-word call were tried.
- **Reading:** each policy is a curve of calls against missed. The curves are read at 20, 35 and 50 % of the calls, next to random skipping (missed = 100 − calls) and an oracle (calls only where a call would deliver). Calls and missed are measured at the same position. Intervals are paired bootstrap over test messages.

**Missed % at 50 % of the calls** (test half, 95 % interval)

| Granularity | Always-invoke delivers at | Best simple rule | Learned policy (chosen on dev) | Learned minus rule |
|---|---|---|---|---|
| sentence | 13.8 % of positions | min words typed: 13.5 [7–23] | 20.0 [11–29] | +6.4 [−1.3, +11.7] |
| phrase | 29.4 % | min words typed: 38.3 [33–44] | 42.4 [37–47] | +4.1 [−1.1, +9.1] |
| next_word | 29.2 % | min words typed: 41.7 [36–48] | 47.0 [40–53] | +5.2 [−0.1, +9.7] |
| word-level mix | 19.6 % | word start + min letters: 30.5 [24–37] | 36.8 [29–43] | +6.3 [−1.1, +13.0] |
| partial_word | 16.3 % | min letters in word: 35.4 [28–45] | 43.8 [34–53] | +8.4 [−3.8, +19.6] |

A positive last column means the simple rule is better. For comparison, random skipping misses 50 % at 50 % of the calls, and the pause rule in the mix misses 54.4 %.

**What the thresholds look like** (test half, from `results/learned_policy_curves.csv`)

| Sentence: calls once m words are typed | calls | missed |
|---|---|---|
| m = 8 | 55.3 % | 13.4 % |
| m = 9 | 48.8 % | 13.5 % |
| m = 10 | 44.7 % | 20.0 % |
| m = 11 | 39.2 % | 23.2 % |
| m = 12 | 35.5 % | 31.4 % |

| Word-level mix: word start + n letters | calls | missed |
|---|---|---|
| n = 2 | 79.8 % | 12.5 % |
| n = 3 | 54.7 % | 26.5 % |
| n = 4 | 43.7 % | 35.8 % |
| n = 5 | 38.1 % | 44.2 % |

| partial_word: calls once the word has n letters | calls | missed |
|---|---|---|
| n = 2 | 73.0 % | 20.0 % |
| n = 3 | 39.5 % | 42.5 % |
| n = 4 | 24.8 % | 57.3 % |

**What the results show**

| Observation | Numbers |
|---|---|
| Pauses do not help | pause rules miss 54.4 % in the mix at 50 % of the calls, worse than random (50 %); a learned model using only timing misses 46.0 % |
| How much has been typed does | sentence: words typed; mid-word: letters in the word. A model using only prefix length reaches 20.0 % for sentence (the full model 19.5 %); without the length inputs it falls to 45.0 % |
| The learned policy does not beat the best simple rule | the simple rule's average is better in every row of the table; most differences lie within the intervals, so the safe statement is that learning adds nothing over a threshold on these inputs |
| A short wait before a mid-word call hurts | in the mix, no wait misses 39.3 %, a 0.4 s wait 45.8 % (logistic regression, 50 % of the calls): the suggestion arrives stale |
| Only sentence meets the proposed targets | `min words typed` passes in 100 % of resamples, at a median of 40 % of the calls [33–47]; the learned policy passes in 88 % |
| Everything else misses the targets | word-level mix: the best rule passes in 6 % of resamples; partial_word 1 %; next_word and phrase never (30–42 % missed at 50 % of the calls) |

**Cautions**

- Only 5 cut points exist per message. Call shares are shares of those points, not of real keystrokes. A words-typed threshold may partly select the late cut points of a message, where less text remains to predict, so the sentence threshold of about 10–11 words should not be taken as a general value yet.
- The test half has been looked at several times. The dev half chose a gradient-boosted model for the mix and partial_word, and on the test half it did no better than the simple rule. A fresh set of messages is needed to confirm.
- Both policy types read their operating point off the test curve, so both carry a similar small optimism.
- A model using only word shape (letters typed, last key a space or letter) reached 29.7 % in the mix, as good as the best rule. It was not chosen on the dev half, so it is a diagnostic only.

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

| # | Rule | Implemented as | Status |
|---|---|---|---|
| 1 | always-invoke (baseline) | call after every keystroke | done |
| 2 | fixed-interval | every 3 and every 5 keystrokes | done |
| 3 | pause-based | call only if no key within 0.2 to 3 s | done |
| 4 | boundary-based | call only right after a space | done |
| 4b | boundary + mid-word pause | call right after every space, and inside a word only after a pause of 0.4 / 0.6 / 1.0 s | done |
| 5 | position-based | call once the word has n letters, or once m words are typed, and combinations | done (§1.4) |
| 6 | learned | logistic regression and small trees on pause and position inputs | done (§1.4) |

## 2.3 The four numbers, always together

| Metric | In the scripts |
|---|---|
| Cost | calls per 100 keystrokes **and** model-seconds per 100 keystrokes (calls × mean latency) |
| Quality | value per call, per `metric_choice.json` |
| Invocation rate | calls per 100 keystrokes (§1.3); in §1.4 the share of decision points at which the model is called, so calls and missed are measured at the same position |
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
                                                          └──► A7  learned + position rules
                                                                  └──► A7b  ceiling ──► A7c  full sessions ──► A8  AESLC
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

`src/policy_eval_v3.py` runs the evaluation of §1.3; it writes `results/policy_eval_v3_summary.csv` (one row per group and rule) for the trade-off chart.

### A7 — Learned policy and position rules
`src/learned_policy.py` compares a learned policy with simple rules on the 240 messages (§1.4). The messages are split once at random into a dev half and a test half. The learned variant is chosen on the dev half, then trained on the whole dev half and scored on the test half, which is never used for a choice. The inputs are known before the call. The label is whether a call delivers a suggestion that is still valid on arrival. The simple rules include the earlier pause rules and the new position rules. Results are read at 20, 35 and 50 % of the calls with 95 % intervals over test messages, and compared with random skipping and an oracle. The script also applies the proposed selection rule and reports in how many bootstrap resamples each policy passes.

Finding: the learned policy adds nothing over a threshold on letters and words typed. Limits: 5 cut points per message, a test half that has been seen several times, and no content-based inputs (§4.2, §4.11, §4.12).

### A7b — Ceiling analysis (next)
Group the decision points by context (word start or mid-word, letters typed, words typed). For each group, count how often a call delivers a still-valid suggestion. The best possible policy for these inputs calls the groups with the highest share first. This gives the best frontier of calls against missed that any policy using these inputs could reach. If the simple rules are already close to it, learning cannot help with these inputs; if there is a gap, it shows where to look. It needs only the existing data.

### A7c — Full typing sessions (after A7b)
Generate suggestions at every keystroke for about 30 messages (single-suggestion timing only), then replay full typing sessions. In a session the model is busy while it generates, and an earlier suggestion can be reused while it still matches what has been typed. The actions are skip, call now, or wait for a pause. The simple rules and a router that picks among them from the session state are compared on real call rates. This removes the fixed cut points (§4.2) and is the place where a learned policy can use information a threshold cannot. Content-based inputs (how predictable the current word is from the training vocabulary, whether the last suggestions in the session were right) are added only if the sessions show room. It runs on the machine with Qwen, and the results are confirmed on about 100 new messages.

## 3.3 Verification

| # | Check | Expected | Status |
|---|---|---|---|
| 1 | keystroke rate with zero delay (v2 file) | 9.9 / 25.9 / 7.1 / 3.9 % | ✅ matches exactly |
| 2 | metric read from `metric_choice.json` | no "could not read" warning | ✅ none in the v3 run |
| 3 | seeded | same output twice | by construction (seed 42; every generation seeded) |
| 4 | v3 contains the v2 messages | all v2 next_word prefixes reappear | ✅ 382 of 382 |
| 5 | pilot re-run independently by a teammate | same rows | ✅ every-3, every-5 and pause 0.6 s rows identical on the v2 file |
| 6 | all typing models run | tables for aalto and styles 0, 1, 2 | aalto ✅ (v2 file: all four) |
| 7 | learned policy on data with no signal | no better than random | ✅ on a toy set |
| 8 | selection-rule check on constructed curves | passes and fails where expected | ✅ |
| 9 | `pytest tests/` | green incl. new tests | tests not written yet |

## 3.4 Reproduce

```bash
uv run python make_pace_table.py             # needs the local AmazonQAC cache (~60 GB)
uv run python fit_aalto_gaps.py ~/Downloads/Keystrokes/files 500   # needs the Aalto download
uv run python -m src.run_eval_v3             # 240 messages, about 2.4 h; resumes if stopped
uv run python -m src.policy_eval_v3 aalto    # the results of section 1.3
uv run python -m src.policy_eval_v3 1        # robustness: medium modelled typist
uv run python -m src.learned_policy          # the results of section 1.4, about 30 min
uv run python -m src.policy_eval 1           # earlier 80-message pilot on phase1_results_v2.csv
```

---

# Part 4 — Open Issues

## 4.1 Gap shape: real, but from copy-typing
The pause-based conclusions rest on the measured gap shape, which lies between the steady and medium modelled styles. See §4.7 for the remaining limit.

## 4.2 Cut points sit at fixed positions
Each message has 5 cut points per granularity. The next_word, phrase and sentence examples are cut at word starts; the partial_word examples inside a word. A boundary rule therefore cannot miss a word-start example on this data, and cannot serve a mid-word one. The word-level mix evaluates both position types with the weights of real typing (about 3.9 : 1, estimated from word lengths, not measured). Because only a few positions exist per message, the call shares in §1.3 and §1.4 are shares of those points, not of real keystrokes, and a threshold on words typed may partly select later positions. Stage A7c evaluates every keystroke. **Feedback required (question 3).**

## 4.3 Invocation rate and value are measured at different positions
In §1.3, calls per 100 keystrokes averages over every keystroke before the cut, while value is measured only at the cut. In styles with pauses after spaces, cut points have longer gaps than the average keystroke, which flatters pause rules in "value kept". Value per call is the safer comparison. §1.4 avoids this by measuring calls and missed at the same position.

## 4.4 Staleness for the similarity-scored granularity
For sentence, a suggestion counts only if its first typed characters match literally, which is strict for paraphrases. With the corrected latency the effect is small, but the rule still needs a decision. **Feedback required (question 4).**

## 4.5 Latency numbers are not production latency
MPS, float32, no prefix cache. Phase 3 (GGUF, with and without caching) will change both the timing and possibly the winner. Every latency is reported with its runtime and with whether it is the first suggestion or five candidates.

## 4.6 AmazonQAC pace includes time after the last keystroke
Duration runs until `search_time`, so per-keystroke pace is slightly overestimated.

## 4.7 Aalto is copy-typing, and the simulated pauses carry no information about content
The Aalto dataset page describes participants who transcribed sentences shown to them, so thinking pauses are rarer than when composing (only 8.2 % of gaps exceed twice the typist's own average). This is why the overall pace comes from AmazonQAC. The dataset's `readme.txt` has not been read yet. In the simulator, a pause depends only on whether the key follows a space; it never says anything about what comes next. In real writing, pauses may come before hard words or at clause boundaries, which could make timing more informative than here. So "pauses do not help" holds under this typing model, not necessarily for real composition. A check on real composition logs (pause length against how predictable the next word is) is open.

## 4.8 Sample size
The v3 run has 240 evaluation messages (the research plan requires more than 200); the test half in §1.4 has 120. The intervals are over messages and do not include the randomness of the simulated typists.

## 4.9 The selection rule is a proposal
The 50 % / 25 % / 0.95 thresholds are not from the research plan. Either the trade-off curve (calls against missed) is reported with the choice left open, or the thresholds are confirmed before policies are compared. **Feedback required (question 1).**

## 4.10 Phase 1 latency in the README
The README latency table (413 / 456 / 1,098 / 3,181 ms) is the five-candidate time; the measured first-suggestion time is 0.11 / 0.10 / 0.29 / 0.72 s. The README is updated once question 2 is settled. **Feedback required (question 2).**

## 4.11 No content-based inputs yet
The inputs of the learned policy are about timing and position. Whether a call pays off depends mostly on whether the model's guess will be right, which these inputs hardly predict. That is why mid-word, next_word and phrase suggestions still miss 30–42 % at 50 % of the calls. Content-based inputs are part of A7c.

## 4.12 The test half has been seen several times
Choices after the first look at the test half (new candidates, new baseline rules) make its numbers exploratory. Confirmation needs new messages that nobody has looked at (about 100, generated with `run_eval_v3.py`), and the AESLC run as a second dataset.

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
| `src/policy_eval_v3.py` | the results of §1.3: timing rules, intervals over messages, word-level mix, PASS column |
| `src/learned_policy.py` | the results of §1.4: position rules, learned policy, dev/test split, PASS check, feature check |
| `phase1_results_v3.csv` | results of the v3 run (the raw `.jsonl` is not committed) |
| `results/` | saved outputs: pilot (aalto, styles 0/1/2), `policy_eval_v3_aalto.txt`, `policy_eval_v3_summary.csv`, `learned_policy_output.txt`, `learned_policy_summary.csv`, `learned_policy_curves.csv` |
| `inspect_aalto.py` | inspects an Aalto zip without unzipping (not needed for the folder download) |
| `phase1_results_v2.csv` | frozen Phase 1 results, read-only; hand labels belong to it |
