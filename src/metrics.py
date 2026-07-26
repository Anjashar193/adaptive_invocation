"""
Step 3: Metrics beyond exact-match. ChaI-TeA's own authors note exact-match likely
under-represents real acceptance behavior - a user often accepts a suggestion that's
close-but-not-identical. These metrics give you graded signal instead of a single
strict 0/1.

Note: semantic_similarity() uses sentence-transformers if available, and falls back
to a cheap lexical-overlap proxy if the package/model can't be loaded (e.g. no
network). Swap in your own embedding model if you already have one loaded.
"""

import difflib
import re

_ST_MODEL = None


def _get_embedding_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _ST_MODEL = False  # mark as unavailable, use fallback
    return _ST_MODEL


def normalize(text):
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def first_word_match(prediction, reference):
    """Does the first predicted word match the first reference word? Much less
    strict than full exact-match, and closer to what actually matters for
    next-word suggestions."""
    pred_words = normalize(prediction).split()
    ref_words = normalize(reference).split()
    if not pred_words or not ref_words:
        return 0.0
    return 1.0 if pred_words[0] == ref_words[0] else 0.0


def partial_word_char_match(prediction, reference_remainder):
    """For partial-word completion: character-level overlap with the correct
    remainder, using longest common prefix (most relevant for autocomplete -
    a suggestion is only useful if its BEGINNING matches)."""
    pred = normalize(prediction)
    ref = normalize(reference_remainder)
    if not ref:
        return 0.0
    match_len = 0
    for a, b in zip(pred, ref):
        if a == b:
            match_len += 1
        else:
            break
    return match_len / len(ref)


def sequence_overlap(prediction, reference):
    """Character-level similarity ratio (cheap, no dependencies) - useful for
    phrase/sentence granularity where exact wording rarely matches but overlap
    still indicates relevance."""
    return difflib.SequenceMatcher(None, normalize(prediction), normalize(reference)).ratio()


def semantic_similarity(prediction, reference):
    """Embedding cosine similarity for phrase/sentence granularity. Falls back
    to sequence_overlap if sentence-transformers isn't available."""
    model = _get_embedding_model()
    if not model:
        return sequence_overlap(prediction, reference)
    import numpy as np
    emb = model.encode([prediction, reference])
    a, b = emb[0], emb[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return cos


def usefulness_score(prediction, reference, full_text, judge_fn=None):
    """
    0 = irrelevant, 1 = close/plausible, 2 = relevant/appendable.
    If judge_fn is provided (e.g. a call to a stronger LLM), use it.
    Otherwise use a heuristic based on semantic similarity as a stand-in
    until you wire up an LLM judge.
    """
    if judge_fn is not None:
        return judge_fn(prediction, reference, full_text)
    sim = semantic_similarity(prediction, reference)
    if sim > 0.75:
        return 2
    elif sim > 0.4:
        return 1
    return 0


def trim_prediction(prediction, reference, granularity):
    """
    Trim a (possibly longer) model prediction down to the same 'unit' the
    reference represents, before computing usefulness. Without this, a
    5-token next-word prediction gets compared against a single reference
    word and is unfairly penalized relative to a baseline that only ever
    predicts one word.
    """
    if granularity == "next_word":
        words = normalize(prediction).split()
        return words[0] if words else ""
    elif granularity == "partial_word":
        return prediction[:len(reference)]
    elif granularity == "phrase":
        ref_len = len(reference.split())
        return " ".join(prediction.split()[:ref_len]) if ref_len else prediction
    elif granularity == "sentence":
        parts = re.split(r'(?<=[.!?]) ', prediction)
        return parts[0] if parts else prediction
    return prediction


def score_example(prediction, example, granularity):
    """Compute the full metric bundle for one example dict from data_prep.py."""
    reference = example["reference"]
    result = {}

    if granularity == "partial_word":
        result["char_match"] = partial_word_char_match(prediction, reference)
    elif granularity == "next_word":
        result["first_word_match"] = first_word_match(prediction, reference)
    else:  # phrase / sentence
        result["semantic_sim"] = semantic_similarity(prediction, reference)
        result["overlap"] = sequence_overlap(prediction, reference)

    trimmed = trim_prediction(prediction, reference, granularity)
    result["usefulness"] = usefulness_score(trimmed, reference, example["full_text"])
    return result


if __name__ == "__main__":
    # sanity checks with obvious cases
    print("first_word_match (should be 1.0):", first_word_match("weather today", "weather is nice"))
    print("first_word_match (should be 0.0):", first_word_match("climate today", "weather is nice"))
    print("partial_word_char_match 'ma' vs 'make' remainder 'ke':",
          partial_word_char_match("keen", "ke"))
    print("sequence_overlap similar sentences:",
          sequence_overlap("the weather is nice today", "the weather looks nice today"))
    print("sequence_overlap unrelated sentences:",
          sequence_overlap("the weather is nice today", "can you write me a poem"))