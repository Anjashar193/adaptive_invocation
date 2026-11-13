"""The invocation policy: when to call the model, and what to ask it for.

This is the frozen result of Phase 2a (see PHASE2A_POLICY_PLAN.md, sections 1.4 and 1.5):
timing does not help, how much has been typed does. The rules are plain thresholds, so
the same function can be used by the evaluation, the llama.cpp runtime (Phase 3) and the
demo server (Phase 4).

    decide(text_before_cursor) -> list of granularities to generate, possibly empty

Rules (each granularity uses the rule that was evaluated for it):
    mid-word      the word being typed has at least MIN_LETTERS letters   -> partial_word
    word start    the last key was a space                                -> next_word
    long message  word start and at least SENTENCE_MIN_WORDS words typed  -> also sentence

"Words typed" is counted as in the evaluation: number of spaces + 1, which includes the
word that is about to start. So SENTENCE_MIN_WORDS = 11 means 10 finished words.

Evaluation numbers behind the defaults (test half, 120 messages, section 1.4; shares of the
evaluated decision points, which are 5 fixed cut points per message, not of all key presses):
    sentence   words typed >= 11: about 39 % of the calls, about 23 % of useful suggestions missed
    word level word start, or mid-word with >= 3 letters: about 55 % of the calls, about 27 % missed
Known limit: this version is stateless. Inside a word it calls again at every further key press
(the demo below calls at about 2/3 of all key presses), although the previous suggestion may still
be valid. Reusing a still-valid suggestion is the obvious next saving (Stage A7c idea).
phrase is not used in this first version: next_word and phrase have the same share of useful
suggestions, and next_word is about three times faster. Choosing between them with measured
value per second is an open refinement (Phase 2b).

Run from the repo root to see the decision after every key of a text:
    uv run python -m src.policy "Could you please recommend some good movies to watch tonight"
"""
import sys
from dataclasses import dataclass

MIN_LETTERS = 3            # letters typed in the current word before a partial_word call
SENTENCE_MIN_WORDS = 11    # words typed (spaces + 1) before a sentence call


@dataclass(frozen=True)
class PolicyParams:
    min_letters: int = MIN_LETTERS
    sentence_min_words: int = SENTENCE_MIN_WORDS


DEFAULT = PolicyParams()


def words_typed(text):
    """Words typed in the sense used by the evaluation: spaces + 1."""
    return text.count(" ") + 1


def letters_in_word(text):
    """Characters typed since the last space."""
    return len(text.split(" ")[-1])


def decide(text_before_cursor, params=DEFAULT):
    """Which granularities to generate after this key press (empty list = do not call)."""
    text = text_before_cursor
    if not text:
        return []
    if text[-1].isspace():                       # word start
        calls = ["next_word"]
        if words_typed(text) >= params.sentence_min_words:
            calls.append("sentence")
        return calls
    # Guard added for real use: the evaluation's cut points were always letters.
    if text[-1].isalpha() and letters_in_word(text) >= params.min_letters:
        return ["partial_word"]
    return []


def replay(text, params=DEFAULT):
    """Decision after every key of `text`: list of (key_index, typed_so_far, calls)."""
    return [(i, text[:i + 1], decide(text[:i + 1], params)) for i in range(len(text))]


def call_rate(text, params=DEFAULT):
    """Share of key presses at which the policy calls the model."""
    steps = replay(text, params)
    return sum(1 for _, _, calls in steps if calls) / len(steps) if steps else 0.0


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "Could you please recommend some good movies to watch tonight after work"
    for i, typed, calls in replay(sample):
        print(f"{i + 1:>3}  {typed[-30:]!r:<34} {', '.join(calls) if calls else '-'}")
    print(f"\ncalls at {100 * call_rate(sample):.0f} % of the {len(sample)} key presses")