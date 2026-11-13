"""Second dataset for Phase 2 (Stage A8): AESLC workplace emails (Enron), as in PROJECT_GUIDELINE.md.

Dataset: Yale-LILY/aeslc on Hugging Face, validation split, field `email_body`.
Filter (the guideline's description, written out as code):
  * 8-60 words after whitespace is collapsed to single spaces,
  * no forwarded/quoted mail headers (From:, To:, Cc:, Sent:, Subject:, "original message"),
  * no signature block (phone or fax number, extension, e-mail address, web address),
  * not list-heavy (many bullet, numbered or very short lines),
  * mostly plain words (at least 85 % alphabetic) with at least one sentence ending.
The guideline reports 657 emails surviving its filter; this filter is a re-implementation, so
the count may differ a little. The script prints the count it gets.

Messages are shuffled with a fixed seed and returned as single-line strings, so they go through
build_eval_set() exactly like the OASST2 messages.

Check it before the long run (prints counts and three examples):
    uv run python -m src.aeslc_prep
"""
import random
import re
import statistics

MIN_WORDS, MAX_WORDS = 8, 60
SEED = 42
HEADER = re.compile(r"(?im)^\s*(from|to|cc|bcc|sent|subject)\s*:|original message|forwarded by|-----")
SIGNATURE = re.compile(r"(\(\d{3}\)|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\bfax\b|\bext\.?\s*\d|@|www\.|http)", re.I)
BULLET = re.compile(r"^([-*•]|\d+[.)])\s")
PLAIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’,.;:!?()\"-]*")


def normalize(body):
    return " ".join(str(body).split())


def is_prose(body):
    """True if the raw email body is a believable prose message (see module docstring)."""
    text = normalize(body)
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if HEADER.search(body) or SIGNATURE.search(body):
        return False
    lines = [ln.strip() for ln in str(body).splitlines() if ln.strip()]
    if len(lines) >= 4:
        bullets = sum(1 for ln in lines if BULLET.match(ln))
        short = sum(1 for ln in lines if len(ln.split()) < 5)
        if bullets / len(lines) > 0.3 or short / len(lines) > 0.5:
            return False
    plain = sum(1 for w in words if PLAIN_WORD.fullmatch(w))
    if plain / len(words) < 0.85:
        return False
    return bool(re.search(r"[.!?]", text))


def select_messages(bodies, max_messages, seed=SEED):
    """Filter, normalise, shuffle with a fixed seed, return the first max_messages."""
    kept = [normalize(b) for b in bodies if is_prose(b)]
    random.Random(seed).shuffle(kept)
    return kept[:max_messages], len(kept)


def _load_bodies():
    from datasets import load_dataset
    errors = []
    for kwargs in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset("Yale-LILY/aeslc", split="validation", **kwargs)
            return [row["email_body"] for row in ds if row.get("email_body")]
        except Exception as e:  # script-based dataset may be refused by newer `datasets`
            errors.append(f"{kwargs or 'default'}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("Could not load Yale-LILY/aeslc:\n  " + "\n  ".join(errors))


def load_aeslc(max_messages=140):
    """Drop-in replacement for data_prep.load_oasst2_prompter_en()."""
    bodies = _load_bodies()
    messages, n_kept = select_messages(bodies, max_messages)
    print(f"AESLC validation: {len(bodies)} emails, {n_kept} pass the filter, using {len(messages)}")
    return messages


if __name__ == "__main__":
    bodies = _load_bodies()
    msgs, n_kept = select_messages(bodies, 10**9)
    words = [len(m.split()) for m in msgs]
    print(f"{len(bodies)} emails in the validation split; {n_kept} pass the filter "
          f"(the guideline reports 657); median {statistics.median(words):.0f} words")
    for m in msgs[:3]:
        print("\n---", m)