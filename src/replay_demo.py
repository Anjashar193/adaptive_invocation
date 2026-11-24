"""Phase 4 measurement: type whole messages into the demo's /suggest logic, one character at a time.

This gives model calls per key press for real texts (every key press of a message, not 5 cut points).
It uses the same code as the demo server (demo/server.py), so it needs the llama.cpp server:
    llama-server -m <path>/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 -c 2048
Then, from the repo root:
    uv run python -m src.replay_demo                    # 60 OASST2 messages, about 10 min
    MESSAGES=240 uv run python -m src.replay_demo       # all evaluation messages, about 45 min
    DATASET=aeslc uv run python -m src.replay_demo      # AESLC emails

Each message is typed in four modes:
    always           a model call at every key press
    always_reuse     a model call at every key press unless the last suggestion still matches
    policy_no_reuse  the frozen rule (src/policy.py), stateless, as evaluated in Phase 2a
    policy           the frozen rule, plus reuse of the last suggestion while it still matches
For every key press it records whether the model was called, and whether a suggestion was on screen
that matches the text the message really continues with ("correct").

Limits: the text is typed without typos, corrections or accepted suggestions, and as fast as the
server answers, so a suggestion never arrives late (no staleness). One sampled suggestion per call.
Writes results/replay_demo_<dataset>.csv (one row per message and mode) and prints a summary.
Environment: MESSAGES (default 60), DATASET (oasst2 or aeslc), MAX_CHARS (default 400, longer messages are cut)
"""
import csv
import os
import random
import sys
import time

from demo import server

DATASET = os.environ.get("DATASET", "oasst2")
MESSAGES = int(os.environ.get("MESSAGES", "60"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "400"))
N_TRAIN = 20                    # the first 20 messages fitted the baselines in Phase 1 and are skipped there too
MODES = ["always", "always_reuse", "policy_no_reuse", "policy"]
N_BOOT = 1000
FIELDS = ["key_presses", "model_calls", "reused", "skipped", "shown", "correct", "model_ms"]


def load_messages():
    if DATASET == "aeslc":
        from src.aeslc_prep import load_aeslc
        msgs = load_aeslc(N_TRAIN + MESSAGES)
    else:
        from src.data_prep import load_oasst2_prompter_en
        msgs = load_oasst2_prompter_en(max_messages=N_TRAIN + MESSAGES)
    return list(enumerate(msgs))[N_TRAIN:]


def type_message(mid, text, mode):
    session = f"replay-{DATASET}-{mid}"
    server.reset(session)
    c = dict.fromkeys(FIELDS, 0)
    for i in range(1, len(text) + 1):
        r = server.suggest(text[:i], session, mode)
        if r["source"] == "error":
            sys.exit(r["error"] + "\nStart llama-server first (see the top of this file).")
        c["key_presses"] += 1
        c["model_calls"] += r["source"] == "model"
        c["reused"] += r["source"] == "reuse"
        c["skipped"] += r["source"] == "skip"
        c["model_ms"] += r["latency_ms"]
        if r["ghost"]:
            c["shown"] += 1
            c["correct"] += text[i:].startswith(r["ghost"])
    server.reset(session)
    return c


def pct(rows, num, den="key_presses"):
    d = sum(r[den] for r in rows)
    return 100 * sum(r[num] for r in rows) / d if d else float("nan")


def interval(rows, num, den="key_presses", rng=random.Random(42)):
    vals = []
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        vals.append(pct(sample, num, den))
    vals.sort()
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def main():
    msgs = load_messages()
    msgs = [(mid, " ".join(t.split())[:MAX_CHARS]) for mid, t in msgs]       # single spaces, as typed text
    n_keys = sum(len(t) for _, t in msgs)
    print(f"Dataset: {DATASET}; {len(msgs)} messages, {n_keys} key presses per mode; llama.cpp server {server.LLAMA_URL}")
    print(f"Frozen policy: mid-word from {server.policy.MIN_LETTERS} letters; sentence from {server.policy.SENTENCE_MIN_WORDS} words")
    out = []
    for mode in MODES:
        t0 = time.time()
        for mid, text in msgs:
            c = type_message(mid, text, mode)
            out.append(dict(c, dataset=DATASET, mode=mode, message_id=mid, model_ms=round(c["model_ms"], 1)))
        print(f"  {mode}: done in {time.time() - t0:.0f} s", flush=True)

    os.makedirs("results", exist_ok=True)
    path = f"results/replay_demo_{DATASET}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "mode", "message_id"] + FIELDS)
        w.writeheader()
        w.writerows(out)

    print()
    print("Shares of key presses, % [95 % interval over messages]")
    print(f"{'mode':<18}{'model calls':>18}{'reused':>10}{'suggestion shown':>20}{'correct shown':>18}{'ms per key':>12}{'ms per call':>13}")
    base = [r for r in out if r["mode"] == "always"]
    for mode in MODES:
        rows = [r for r in out if r["mode"] == mode]
        lo, hi = interval(rows, "model_calls")
        clo, chi = interval(rows, "correct")
        calls = sum(r["model_calls"] for r in rows)
        ms = sum(r["model_ms"] for r in rows)
        print(f"{mode:<18}"
              + f"{pct(rows, 'model_calls'):.1f} [{lo:.0f}-{hi:.0f}]".rjust(18)
              + f"{pct(rows, 'reused'):.1f}".rjust(10)
              + f"{pct(rows, 'shown'):.1f}".rjust(20)
              + f"{pct(rows, 'correct'):.1f} [{clo:.0f}-{chi:.0f}]".rjust(18)
              + f"{ms / max(1, sum(r['key_presses'] for r in rows)):.1f}".rjust(12)
              + f"{ms / max(1, calls):.1f}".rjust(13))
    print()
    print("correct shown = a suggestion is on screen and equals the next characters of the message.")
    print("Each mode samples its own suggestions, so small differences in 'correct shown' are noise.")
    print("With reuse, a matching suggestion stays on screen while it is typed out, which raises 'correct shown'.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
