# Project Guideline

**The goal:** show that a lightweight adaptive policy can cut wasted model calls without losing useful suggestions. Phase 2 is the research; everything else supports it.

**Rule:** finish a phase's exit criteria before starting the next. No building ahead.

---

## Phase 1 — Generation quality baseline — *DONE*

Qwen3-0.6B vs. frequency/n-gram/random baselines across 4 granularities on OASST2.
Files: `data_prep.py`, `baselines.py`, `metrics.py`, `run_eval.py`, `recompute_metrics.py`, `report.py`.

**Cleanup — done:**
- [x] Made `semantic_similarity()` fail loudly if `sentence-transformers` is missing (no silent difflib fallback)
- [x] Re-ran `run_eval.py` (MPS), regenerated CSVs — usefulness numbers unchanged, confirms old CSVs already used real embeddings
- [x] Updated README — latency numbers now labeled with device/precision (MPS, not CPU as before); usefulness numbers didn't move so no other changes needed

**Known minor issue, not blocking:** `run_eval.py` saves its CSV relative to the current working directory, so the README's copy-paste commands only produce the right file location if run from `src/`.

---

## Phase 2 — Invocation policy *(the contribution)*

This is the novel part. It comes before any runtime or production work.

**2.0. Validate the thresholds first.** Hand-label a few hundred examples and check the 0.75/0.4 usefulness thresholds against them. Phase 2's success is measured with these thresholds — if they're wrong, everything after is wrong. Do this before anything else in Phase 2.

**2a. When to invoke.** Build the typing-simulation layer first. **Don't invent pause timings** — take them from a real keystroke/prefix-typing dataset and record the source. Primary source: **Amazon Query Autocomplete (`amazon/AmazonQAC`)** — real prefix-typing sequences with `first_prefix_typed_time`/`search_time`, i.e. actual inter-prefix pause behavior during autocomplete, not simulated. Confirm whether timestamps are per-prefix or per-keystroke before relying on them for fine-grained pause modeling; fall back to the Aalto typing studies if finer granularity is needed. Then compare: always-invoke (baseline) → fixed-interval → pause-based → boundary-based → learned/adaptive.

**2b. How much to generate.** Use Phase 1's per-granularity quality/latency to pick partial-word / next-word / phrase / sentence per invocation, instead of always running all four.

**Evaluation — define once, reuse everywhere:**
- **Cost = number of model calls + their latency.** Both, always, stated explicitly next to every result.
- Primary: usefulness per invocation cost (reuse `metrics.usefulness_score` — don't rewrite)
- Secondary: invocation rate (% of keystrokes that triggered a call)
- Third: **missed opportunities** — of the good suggestions always-invoke found, how many did the policy skip? Without this, a policy that rarely calls looks better than it is.
- **Staleness rule:** a suggestion only counts if it's still correct when it arrives. A sentence takes ~1.9s to generate; by then the user has typed more. Late-and-wrong = failure.
- **Always report all of the above together.** Any one alone is meaningless.

**Learned policy rules:**
- Each message gives 5 examples (5 cut points). All 5 must land on the same side of the train/test split — never split by example.
- Use more than 200 source messages. OASST2 has plenty; scaling is cheap.

**Datasets:** develop on OASST2 only. Once the policy works there, run it once on the second dataset: **Enron emails (`Yale-LILY/aeslc` on Hugging Face, validation split, `email_body` field), filtered to 8–60 words and with signature-block/list-heavy emails dropped (657 emails survive).** These two cover completion *quality* across all four granularities (they're raw prose the harness can cut at arbitrary word ratios). They do **not** cover invocation *timing* — for that, Phase 2a uses **AmazonQAC** (see below) as a third, timing-only dataset. The filter matters: raw `email_body` has a median of 77 words vs. OASST2's 17, and many emails are greetings/sign-offs/line-item lists rather than prose — cutting those at 15/30/50/70/85% would land inside boilerplate, not natural text. Filtered, the median is 43 words, close to OASST2's p90 (48) — a believable "longer message," not a different genre. Real workplace emails, unpolished, a different style from OASST2's assistant prompts, and the standard test set in autocomplete research (Smart Compose). A paper needs this; development doesn't.

**Exit criteria:** a policy that measurably lowers invocation rate vs. always-invoke while keeping usefulness above the hand-validated threshold, with missed opportunities and staleness reported, confirmed on the second dataset. Write it up — this is the publishable result.

---

## Phase 3 — Production runtime validation

Only after the Phase 2 policy is fixed. `run_eval.py`'s full-precision PyTorch/CPU latency isn't production-representative.

**3a. CPU — llama.cpp + GGUF.** Convert Qwen3-0.6B to GGUF (test Q4_K_M and Q8_0 against the usefulness threshold). Swap `generate_completion()`'s internals for a llama.cpp backend, same signature. Re-run the Phase 2 comparison — confirm the savings hold.

**Test caching (with and without).** When the user types one more letter, llama.cpp can reuse the previous call's work instead of starting over (prefix/KV cache). This makes repeat calls much cheaper — and cheap calls may change which policy wins. Run the Phase 2 comparison both ways.

**3b. Mobile — LiteRT-LM.** Export per [LiteRT-LM docs](https://developers.google.com/edge/litert-lm), benchmark on device/emulator, same re-validation. **Measure memory too, not just speed** — a fast model that keeps 600MB resident may be unusable on a phone.

**Exit criteria:** latency + memory table for PyTorch vs. GGUF (cached and uncached) vs. LiteRT, policy re-confirmed on each.

---

## Phase 4 — Demo (server + Next.js)

Last, and only once Phase 2 + 3 hold. Demo a proven policy — don't build ahead of it.
- Server: one `/suggest` endpoint wrapping the Phase 3 runtime + Phase 2 policy module directly (no reimplementation).
- Frontend: Next.js input box with ghost-text shown only when the policy triggers, labeled with the granularity it picked.

---

## Standing rules (every phase)

- **Publishability:** name dataset construction precisely (already in README). Keep documenting bugs in the README as you find/fix them — this is already a strength. Record the runtime + precision + cache setting behind every latency number; never let it silently vary.
- **Verification:** re-run end to end, keep the same CSV/report format across phases, and always show new results side by side with Phase 1's always-invoke baseline. Phase 4 verification is a manual browser test — type, confirm suggestions appear only on policy triggers.