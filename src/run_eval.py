"""
Step 4: The eval harness. This ties together data_prep, baselines, and metrics,
and runs YOUR Qwen3-0.6B alongside the baselines so every result has a floor
to compare against.

Wire in your existing generation code where marked TODO - this script assumes
you already have a `generate_completion(prefix, granularity)` function from your
current pipeline (the one you used for the 9/6/4/20-example manual eval).
"""

import csv
import time
from data_prep import load_oasst2_prompter_en, build_eval_set
from baselines import FrequencyBaseline, NgramBaseline, RandomBaseline
from metrics import score_example

GRANULARITIES = ["partial_word", "next_word", "phrase", "sentence"]
N_MESSAGES = 200          # -> ~1000 examples total across split ratios, plenty for signal
TOKEN_BUDGET = {"partial_word": 5, "next_word": 5, "phrase": 15, "sentence": 40}

# --- Model setup (loaded once, at import time) ---
# NOTE: swap MODEL_NAME / loading code here if your original manual-eval script
# used a different checkpoint path, vLLM, or a pipeline() object instead.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"  # base model, matching your earlier manual eval

FORCE_CPU = True  # set False to allow MPS/CUDA acceleration if you want faster runs later
if FORCE_CPU:
    _device = "cpu"
else:
    _device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(_device)
_model.eval()


def generate_completion(prefix, granularity):
    """Generate a continuation for `prefix`, budgeted by granularity."""
    max_tokens = TOKEN_BUDGET[granularity]
    inputs = _tokenizer(prefix, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,          # greedy - deterministic, matches your earlier setup
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    generated = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def run_full_eval(output_csv="phase1_results.csv"):
    print("Loading data...")
    messages = load_oasst2_prompter_en(max_messages=N_MESSAGES)
    train_texts = messages[: len(messages) // 5]   # small slice reserved to fit baselines
    eval_messages = messages[len(messages) // 5:]

    print("Fitting baselines...")
    freq = FrequencyBaseline(); freq.fit(train_texts)
    ngram = NgramBaseline(); ngram.fit(train_texts)
    rnd = RandomBaseline(); rnd.fit(train_texts)

    rows = []
    for granularity in GRANULARITIES:
        examples = build_eval_set(eval_messages, granularity)
        print(f"[{granularity}] {len(examples)} examples")

        for i, ex in enumerate(examples):
            if i % 25 == 0:
                print(f"  {granularity}: {i}/{len(examples)}", flush=True)
            prefix = ex["prefix"]

            candidates = {
                "qwen": None,   # filled below
                "frequency_baseline": freq.predict_next_word(prefix),
                "ngram_baseline": ngram.predict_next_word(prefix),
                "random_baseline": rnd.predict_next_word(prefix),
            }

            # Qwen generation with latency measurement
            t0 = time.time()
            try:
                candidates["qwen"] = generate_completion(prefix, granularity)
            except NotImplementedError:
                candidates["qwen"] = ""  # allows script to run end-to-end before wiring model in
            latency_ms = (time.time() - t0) * 1000

            for system_name, prediction in candidates.items():
                scores = score_example(prediction, ex, granularity)
                row = {
                    "granularity": granularity,
                    "system": system_name,
                    "prefix": prefix,
                    "reference": ex["reference"],
                    "prediction": prediction,
                    "latency_ms": latency_ms if system_name == "qwen" else 0,
                }
                row.update(scores)
                rows.append(row)

    # write CSV
    if rows:
        # Different granularities produce different metric columns
        # (char_match / first_word_match / semantic_sim+overlap), so we need
        # the union of all keys across all rows, not just rows[0].
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            # restval="" fills in blanks for rows missing a given granularity's
            # metric column (e.g. a next_word row has no semantic_sim value)
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_csv}")

    return rows


if __name__ == "__main__":
    run_full_eval()