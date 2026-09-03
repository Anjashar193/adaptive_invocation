import csv
import time
import re
from data_prep import load_oasst2_prompter_en, build_eval_set
from baselines import FrequencyBaseline, NgramBaseline, RandomBaseline
from metrics import score_example, topk_score

GRANULARITIES = ["partial_word", "next_word", "phrase", "sentence"]
N_MESSAGES = 100
TOP_K = 5
TOPK_GRANULARITIES = {"next_word", "partial_word", "phrase", "sentence"}

# Safety ceilings only - actual stopping is now handled by GranularityStop below,
# so generation stops as soon as the granularity's real target is reached
# (a completed word, 5 words, or a full sentence) rather than overshooting.
MAX_TOKEN_CEILING = {"partial_word": 10, "next_word": 10, "phrase": 25, "sentence": 60}

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL_NAME = "Qwen/Qwen3-0.6B"

FORCE_CPU = False
if FORCE_CPU:
    _device = "cpu"
else:
    _device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(_device)
_model.eval()


class GranularityStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, granularity):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.granularity = granularity

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        stripped = text.strip()
        if not stripped:
            return False

        if self.granularity in ("partial_word", "next_word"):
            words = stripped.split()
            if len(words) >= 2:
                return True
            return bool(re.search(r"[,.!?;:]", stripped))
        elif self.granularity == "phrase":
            return len(stripped.split()) >= 5
        elif self.granularity == "sentence":
            return bool(re.search(r"[.!?](\s|$)", text))
        return False


def generate_topk(prefix, granularity, k=TOP_K):
    inputs = _tokenizer(prefix, return_tensors="pt").to(_device)
    prompt_len = inputs["input_ids"].shape[1]
    max_tokens = MAX_TOKEN_CEILING[granularity]

    candidates = []
    seen = set()
    for _ in range(k):
        stopping = StoppingCriteriaList([GranularityStop(_tokenizer, prompt_len, granularity)])
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                stopping_criteria=stopping,
                pad_token_id=_tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][prompt_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if granularity in ("partial_word", "next_word") and text:
            # trim off the "peek" word that was only needed to detect the boundary
            first_word = text.split()[0] if text.split() else text
            text = first_word

        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    return candidates if candidates else [""]


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

        for i, ex in enumerate(examples):
            if i % 25 == 0:
                print(f"  {granularity}: {i}/{len(examples)}", flush=True)
            prefix = ex["prefix"]

            t0 = time.time()
            qwen_topk = generate_topk(prefix, granularity, k=TOP_K)
            latency_ms = (time.time() - t0) * 1000

            if granularity in ("phrase", "sentence"):
                candidates = {
                    "qwen": qwen_topk,
                    "frequency_baseline": freq.predict_top_k_chain(prefix, granularity, k=TOP_K),
                    "ngram_baseline": ngram.predict_top_k_chain(prefix, granularity, k=TOP_K),
                    "random_baseline": rnd.predict_top_k_chain(prefix, granularity, k=TOP_K),
                }
            else:
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