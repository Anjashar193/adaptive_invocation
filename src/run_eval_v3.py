"""Phase 1 re-run at scale for Phase 2a (Stage A6). Writes NEW files; phase1_results_v2.csv is never touched.

Differences to run_eval.py:
  * 260 messages are loaded; the first 20 fit the baselines (same as the v2 run),
    the other 240 are evaluated (the 80 v2 messages are among them).
  * every row carries message_id, cut_index, prefix_words, msg_words, so the
    train/test split of a learned policy can be done BY MESSAGE.
  * latency_top1_ms = time until the first suggestion exists (what a user waits for);
    latency_ms = time for all 5 sampled candidates (as in v2).
  * every example is seeded -> the same run gives the same predictions.
  * results are appended per message to a .jsonl file, so the run can be stopped
    (Ctrl+C) and restarted; at the end the .csv is written.

Run from the repo root:
    LIMIT=2 OUT=smoke_v3 uv run python -m src.run_eval_v3     # quick test, 2 messages
    caffeinate -i uv run python -m src.run_eval_v3             # full run (about 2-3 hours)
"""
import csv
import json
import os
import random
import time

try:  # works both as `python src/run_eval_v3.py` and as `python -m src.run_eval_v3`
    from data_prep import load_oasst2_prompter_en, build_eval_set
    from baselines import FrequencyBaseline, NgramBaseline, RandomBaseline
    from metrics import score_example, topk_score
    import run_eval as base
except ImportError:
    from src.data_prep import load_oasst2_prompter_en, build_eval_set
    from src.baselines import FrequencyBaseline, NgramBaseline, RandomBaseline
    from src.metrics import score_example, topk_score
    import src.run_eval as base

GRANULARITIES = ["partial_word", "next_word", "phrase", "sentence"]
N_MESSAGES = int(os.environ.get("N_MESSAGES", "260"))
N_TRAIN = 20                      # same baseline-fitting set as the v2 run
SEED = 42
TOP_K = 5
LIMIT = int(os.environ.get("LIMIT", "0"))      # 0 = all evaluation messages
OUT = os.environ.get("OUT", "phase1_results_v3")


def generate_topk_timed(prefix, granularity, seed, k=TOP_K):
    """Same sampling as run_eval.generate_topk, seeded, with the time to the first suggestion."""
    import torch
    from transformers import StoppingCriteriaList
    torch.manual_seed(seed)
    tok, model, dev = base._tokenizer, base._model, base._device
    inputs = tok(prefix, return_tensors="pt").to(dev)
    prompt_len = inputs["input_ids"].shape[1]
    candidates, seen = [], set()
    t0 = time.time()
    t_first = None
    for _ in range(k):
        stopping = StoppingCriteriaList([base.GranularityStop(tok, prompt_len, granularity)])
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=base.MAX_TOKEN_CEILING[granularity],
                                 do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                                 stopping_criteria=stopping, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        if granularity in ("partial_word", "next_word") and text:
            text = text.split()[0]
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)
            if t_first is None:
                t_first = time.time() - t0
    total = time.time() - t0
    return (candidates or [""]), (t_first if t_first is not None else total) * 1000, total * 1000


def finalize(path_jsonl, path_csv):
    done, rows = set(), {}
    with open(path_jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "_done" in r:
                done.add(r["_done"])
            else:
                rows[(r["message_id"], r["granularity"], r["system"], r["cut_index"])] = r
    rows = [r for r in rows.values() if r["message_id"] in done]
    rows.sort(key=lambda r: (r["message_id"], GRANULARITIES.index(r["granularity"]), r["cut_index"]))
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows from {len(done)} messages to {path_csv}")


def main():
    print("Loading data...")
    messages = load_oasst2_prompter_en(max_messages=N_MESSAGES)
    train_texts = messages[:N_TRAIN]
    eval_ids = list(range(N_TRAIN, len(messages)))
    if LIMIT:
        eval_ids = eval_ids[:LIMIT]
    print(f"{len(messages)} messages loaded; baselines fit on {len(train_texts)}; evaluating {len(eval_ids)}")

    print("Fitting baselines...")
    freq = FrequencyBaseline(); freq.fit(train_texts)
    ngram = NgramBaseline(); ngram.fit(train_texts)
    rnd = RandomBaseline(); rnd.fit(train_texts)

    path_jsonl, path_csv = OUT + ".jsonl", OUT + ".csv"
    done = set()
    if os.path.exists(path_jsonl):
        with open(path_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "_done" in r:
                    done.add(r["_done"])
        print(f"Resuming: {len(done)} messages already done")

    t_start, n_new = time.time(), 0
    for mid in eval_ids:
        if mid in done:
            continue
        text = messages[mid]
        lines = []
        for gi, g in enumerate(GRANULARITIES):
            random.seed(SEED * 100003 + mid)               # makes the partial_word split point reproducible
            examples = build_eval_set([text], g)
            for ci, ex in enumerate(examples):
                prefix = ex["prefix"]
                seed = SEED * 1000003 + mid * 100 + gi * 10 + ci
                qwen_topk, lat1, lat5 = generate_topk_timed(prefix, g, seed)
                random.seed(seed)                           # baselines may use `random`
                if g in ("phrase", "sentence"):
                    candidates = {
                        "qwen": qwen_topk,
                        "frequency_baseline": freq.predict_top_k_chain(prefix, g, k=TOP_K),
                        "ngram_baseline": ngram.predict_top_k_chain(prefix, g, k=TOP_K),
                        "random_baseline": rnd.predict_top_k_chain(prefix, g, k=TOP_K),
                    }
                else:
                    candidates = {
                        "qwen": qwen_topk,
                        "frequency_baseline": freq.predict_top_k(prefix, k=TOP_K),
                        "ngram_baseline": ngram.predict_top_k(prefix, k=TOP_K),
                        "random_baseline": rnd.predict_top_k(prefix, k=TOP_K),
                    }
                for system_name, preds in candidates.items():
                    top1 = preds[0] if preds else ""
                    scores = score_example(top1, ex, g)
                    tk = topk_score(preds, ex["reference"], g)
                    is_q = system_name == "qwen"
                    row = {
                        "message_id": mid, "cut_index": ci,
                        "prefix_words": len(prefix.split()), "msg_words": len(text.split()),
                        "granularity": g, "system": system_name,
                        "prefix": prefix, "reference": ex["reference"],
                        "prediction": top1, "topk_predictions": " | ".join(preds),
                        "latency_ms": lat5 if is_q else 0,
                        "latency_top1_ms": lat1 if is_q else 0,
                        "topk_hit": tk["topk_hit"], "topk_rank": tk["topk_rank"],
                    }
                    row.update(scores)
                    lines.append(json.dumps(row))
        lines.append(json.dumps({"_done": mid}))
        with open(path_jsonl, "a", encoding="utf-8") as f:     # one write per message
            f.write("\n".join(lines) + "\n")
        n_new += 1
        per = (time.time() - t_start) / n_new
        left = (len(eval_ids) - len(done) - n_new) * per
        print(f"message {mid} done ({n_new} new this run, ~{per:.0f}s each, ~{left / 60:.0f} min left)", flush=True)

    if os.path.exists(path_jsonl):
        finalize(path_jsonl, path_csv)


if __name__ == "__main__":
    main()