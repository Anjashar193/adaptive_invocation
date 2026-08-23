import csv
import time
from data_prep import load_oasst2_prompter_en, build_eval_set
from baselines import FrequencyBaseline, NgramBaseline, RandomBaseline
from metrics import score_example, topk_score

GRANULARITIES = ["partial_word", "next_word", "phrase", "sentence"]
N_MESSAGES = 200
TOKEN_BUDGET = {"partial_word": 5, "next_word": 5, "phrase": 15, "sentence": 40}
TOP_K = 5
TOPK_GRANULARITIES = {"next_word", "partial_word"}

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

FORCE_CPU = False
if FORCE_CPU:
    _device = "cpu"
else:
    _device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(_device)
_model.eval()


def generate_completion(prefix, granularity):
    max_tokens = TOKEN_BUDGET[granularity]
    inputs = _tokenizer(prefix, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_topk(prefix, granularity, k=TOP_K):
    max_tokens = TOKEN_BUDGET[granularity]
    inputs = _tokenizer(prefix, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            num_return_sequences=k,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            pad_token_id=_tokenizer.eos_token_id,
        )
    candidates = []
    seen = set()
    for seq in output_ids:
        new_tokens = seq[inputs["input_ids"].shape[1]:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        if text not in seen:
            seen.add(text)
            candidates.append(text)
    return candidates


def run_full_eval(output_csv="phase1_results.csv"):
    print("Loading data...")
    messages = load_oasst2_prompter_en(max_messages=N_MESSAGES)
    train_texts = messages[: len(messages) // 5]
    eval_messages = messages[len(messages) // 5:]

    print("Fitting baselines...")
    freq = FrequencyBaseline(); freq.fit(train_texts)
    ngram = NgramBaseline(); ngram.fit(train_texts)
    rnd = RandomBaseline(); rnd.fit(train_texts)

    rows = []
    for granularity in GRANULARITIES:
        examples = build_eval_set(eval_messages, granularity)
        print(f"[{granularity}] {len(examples)} examples")
        use_topk = granularity in TOPK_GRANULARITIES

        for i, ex in enumerate(examples):
            if i % 25 == 0:
                print(f"  {granularity}: {i}/{len(examples)}", flush=True)
            prefix = ex["prefix"]

            if use_topk:
                t0 = time.time()
                qwen_topk = generate_topk(prefix, granularity, k=TOP_K)
                latency_ms = (time.time() - t0) * 1000

                candidates = {
                    "qwen": qwen_topk,
                    "frequency_baseline": freq.predict_top_k(prefix, k=TOP_K),
                    "ngram_baseline": ngram.predict_top_k(prefix, k=TOP_K),
                    "random_baseline": rnd.predict_top_k(prefix, k=TOP_K),
                }

                for system_name, topk_predictions in candidates.items():
                    top1 = topk_predictions[0] if topk_predictions else ""
                    scores = score_example(top1, ex, granularity)
                    tk = topk_score(topk_predictions, ex["reference"], granularity)
                    row = {
                        "granularity": granularity,
                        "system": system_name,
                        "prefix": prefix,
                        "reference": ex["reference"],
                        "prediction": top1,
                        "topk_predictions": " | ".join(topk_predictions),
                        "latency_ms": latency_ms if system_name == "qwen" else 0,
                        "topk_hit": tk["topk_hit"],
                        "topk_rank": tk["topk_rank"],
                    }
                    row.update(scores)
                    rows.append(row)

            else:
                candidates = {
                    "qwen": None,
                    "frequency_baseline": freq.predict_next_word(prefix),
                    "ngram_baseline": ngram.predict_next_word(prefix),
                    "random_baseline": rnd.predict_next_word(prefix),
                }

                t0 = time.time()
                candidates["qwen"] = generate_completion(prefix, granularity)
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

    if rows:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_csv}")

    return rows


if __name__ == "__main__":
    run_full_eval()