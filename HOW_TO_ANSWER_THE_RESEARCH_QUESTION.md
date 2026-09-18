# How to Answer the Research Question

**Read this first. Follow it in order. Do not skip ahead.**

---

## The research question

> Can a lightweight adaptive invocation policy cut unnecessary model calls
> without losing useful suggestions?

To answer it you must show: **fewer model calls, and the suggestions are still
good.** Two halves. You cannot claim the first without measuring the second.

## Why you cannot answer it yet

"The suggestions are still good" is measured by `usefulness_score()`. That
scorer disagrees with a human 41% of the time, and we now know why: it measures
*similarity to what the user typed*, while the human was judging *whether the
text is a sensible thing to write*. Two different questions.

So if you build the policy today, your headline result would be:

> "Our policy keeps usefulness above threshold" — measured by a scorer that
> does not measure usefulness.

An examiner will find that. Fix the ruler before you measure with it.

```
  [ 1. Fix the ruler ]  →  [ 2. Build the policy ]  →  [ 3. Write the paper ]
     ← YOU ARE HERE          the contribution
```

---

# Part 1 — Fix the ruler

## Step 1. Check the code still runs

```bash
uv sync
uv run pytest                     # expect: 36 passed
uv run python compare_label.py    # expect: 299/300 joined
```

If either fails, stop and fix it before labeling anything.

## Step 2. Build the control set

```bash
uv run python -m src.control_set
```

Creates `control_set.csv`: **345 items** (120 partial_word, 75 each of the
others). Sizes come from a power simulation — at 75, partial_word would detect
its own effect only 58% of the time, a coin flip. Do not reduce them.

## Step 3. Label pass 1 — plausibility (reference hidden)

```bash
uv run python hand_label.py --axis plausible --labeler A
```

You see the prefix and one candidate. **You do not see what the user actually
typed.** That is deliberate.

Ask only: **could a competent writer continue this way?**

- `1` = yes, this reads naturally here
- `0` = no, this is broken or nonsensical here

Type `q` any time to save and quit; re-run to resume.

> ⚠ Finish this entire pass before starting Step 4. If you see the reference
> first, it will anchor your judgment and the two axes stop being independent —
> which is exactly the flaw in the last round.

## Step 4. Label pass 2 — match (reference shown)

```bash
uv run python hand_label.py --axis matches --labeler A
```

Now you see what the user actually typed. Ask: **does the candidate agree with
it?**

- `2` = agrees
- `1` = same intent, different words
- `0` = unrelated

## Step 5. Get a second labeler — do not skip this

Find one person (another student is fine). Have them label the **first 100
items only**:

```bash
uv run python hand_label.py --axis plausible --labeler B --limit 100
uv run python hand_label.py --axis matches   --labeler B --limit 100
```

**Why this matters more than anything else in Part 1.** Right now you cannot
tell these two stories apart:

| Story | Meaning |
|---|---|
| "Our scorer is bad" | fixable — improve the metric |
| "The task is ambiguous" | not fixable — humans don't agree either |

Both produce ~59% agreement. If two humans agree only ~70% with each other,
then no metric can beat ~70%, and your scorer is already at the ceiling — the
"bad scorer" story would be **wrong**.

Without labeler B, this stays unresolved and it is the first question an
examiner will ask. A second labeler settles it.

## Step 6. Decide the metric

Implement `src/decide_metric.py`. The rule is **pre-registered** — fix it now,
before you see results, so you cannot talk yourself into a preferred answer:

For each granularity, run the paired bootstrap (`src/agreement.py` already has
it) comparing keystrokes vs embeddings against the preference labels.

| Bootstrap result | Decision |
|---|---|
| CI excludes 0 | adopt the winner |
| CI includes 0 (a tie) | adopt **keystrokes** — simpler, no labels, no thresholds |

> Do **not** tune thresholds and do **not** average the two metrics. Both were
> tested and both made things worse. A tie is a real finding — report it with
> the interval.

Write the outcome to `metric_choice.json`. Phase 2 reads that file. Do not
revisit this decision later.

## Step 7. Amend the guideline

`PROJECT_GUIDELINE.md` §2.0 currently says `usefulness_score` is Phase 2's
primary metric. If Step 6 replaced it, **edit that file now**, or you will
build Phase 2 on the metric you just retired.

✅ **Part 1 is done when `metric_choice.json` exists and the guideline matches it.**

---

# Part 2 — Build the policy (the contribution)

This is your thesis. Everything before it was preparation.

## Step 8. Build the typing simulator

The policy decides *when* to call the model, so you need realistic typing
timing. **Do not invent pause timings.** Use `amazon/AmazonQAC` — real
prefix-typing sequences with timestamps.

First: confirm whether the timestamps are per-keystroke or per-prefix. This
changes what you can model. Write down the answer and cite it.

## Step 9. Compare the policies

Implement these **in order**, simplest first:

| # | Policy | Rule |
|---|---|---|
| 1 | always-invoke | call on every keystroke — **the baseline** |
| 2 | fixed-interval | call every N keystrokes |
| 3 | pause-based | call when the user pauses > T ms |
| 4 | boundary-based | call at word/sentence boundaries |
| 5 | learned | train a small classifier |

Do not skip to 5. If a simple rule wins, that is a **better** paper — it is
cheaper to deploy on-device, which is the whole point.

## Step 10. Report all four numbers, always

Never report one alone. Each is meaningless without the others:

| Metric | Question it answers |
|---|---|
| **Cost** | how many calls, and how much latency? |
| **Quality** | usefulness per call, using `metric_choice.json` |
| **Invocation rate** | what % of keystrokes triggered a call? |
| **Missed opportunities** | of the good suggestions always-invoke found, how many did we skip? |

Without the fourth, a policy that almost never calls looks great. It isn't.

**Staleness rule:** a suggestion only counts if it is still correct when it
arrives. A sentence takes ~1.9s; by then the user has typed more. Late and
wrong = failure. Enforce this in code, not in prose.

## Step 11. Split the data correctly

Each message produces 5 examples (5 cut points). **All 5 must go to the same
side of the train/test split.** Splitting by example leaks — near-identical
prefixes land on both sides and your accuracy is inflated.

Use **more than 200 source messages**:

```bash
N_MESSAGES=260 uv run python src/run_eval.py
```

> ⚠ This overwrites results and invalidates the control set. Only do it
> **after** Step 6 is finished.

## Step 12. Confirm on a second dataset

Once the policy works on OASST2, run it **once** on AESLC (`Yale-LILY/aeslc`,
validation, `email_body`, filtered to 8–60 words, signature/list-heavy emails
dropped — 657 survive). Different genre, real workplace text. A paper needs
this; development does not.

✅ **Part 2 is done when a policy lowers invocation rate vs. always-invoke,
keeps quality above threshold, and holds on AESLC — with all four numbers
reported.**

---

# Part 3 — Write it up

Your paper has **two** contributions. The first came out of Part 1:

> Embedding similarity conflates *plausibility* with *reference-match*. We show
> it ceilings at 68.9% agreement with human judgment, and that keystroke
> savings — a label-free symbolic metric — is better on literal granularities
> but zero-inflates at 63–82% on open-ended ones. Neither alone suffices.

That is publishable on its own and it justifies your evaluation choices. The
second contribution is the policy itself.

**Report honestly.** Ties, negative results, and "the simple rule won" are all
legitimate findings. A tuned number you cannot defend is not.

---

# Rules that apply throughout

1. **Simple before complex.** Try the rule-based version first; only escalate
   when it demonstrably fails. (Your ROARS rule A.)
2. **Never evaluate a system with itself.** No LLM-as-judge unless everything
   else has failed.
3. **Record runtime, precision, and cache setting next to every latency
   number.** Current numbers are MPS float32 — not production. They will change
   in Phase 3.
4. **Keep documenting bugs in the README** as you find them. It is already a
   strength of this project.
5. **Seed everything.** Unseeded results are not reproducible and not
   publishable.

---

# On the multilingual suggestion

From the interim presentation: "the project would be stronger with other
languages."

Adding languages is **breadth, not novelty** — the same policy in German is a
second table, not a second idea. Only Spanish has usable volume anyway (OASST2
validation: es 410, de 121, fr 70 messages).

There *is* a real version of the argument. Tokenization efficiency differs by
language, which changes per-call latency, which is the direct input to your
policy. So:

> "Agreed that language matters, but as a **stress test** rather than coverage.
> After the policy works in English, we re-run on Spanish to test whether
> tokenization-driven latency differences shift the optimal invocation
> threshold. If they do, that strengthens the contribution; if not, it is
> evidence the policy is language-robust."

Do this **after** Part 2. Never replicate a broken measurement in two languages.

---

# Quick reference

```bash
uv run pytest                                          # 36 tests
uv run python compare_label.py                         # agreement analysis
uv run python -m src.control_set                       # build control_set.csv
uv run python hand_label.py                            # show labeling progress
uv run python hand_label.py --axis plausible --labeler A
uv run python hand_label.py --axis matches   --labeler A
```

| File | What it is |
|---|---|
| `PHASE2_METRIC_PLAN.md` | full analysis and evidence behind Part 1 |
| `PROJECT_GUIDELINE.md` | the phase roadmap |
| `control_set.csv` | 345 items to label (regenerate any time) |
| `control_labels.csv` | your labels — **this one is precious, commit it** |
| `phase1_results_v2.csv` | frozen Phase 1 results — do not overwrite |
