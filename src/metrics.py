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
            _ST_MODEL = False
    return _ST_MODEL


def normalize(text):
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def first_word_match(prediction, reference):
    pred_words = normalize(prediction).split()
    ref_words = normalize(reference).split()
    if not pred_words or not ref_words:
        return 0.0
    return 1.0 if pred_words[0] == ref_words[0] else 0.0


def partial_word_char_match(prediction, reference_remainder):
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
    return difflib.SequenceMatcher(None, normalize(prediction), normalize(reference)).ratio()


def semantic_similarity(prediction, reference):
    model = _get_embedding_model()
    if not model:
        return sequence_overlap(prediction, reference)
    import numpy as np
    emb = model.encode([prediction, reference])
    a, b = emb[0], emb[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def usefulness_score(prediction, reference, full_text, judge_fn=None):
    if judge_fn is not None:
        return judge_fn(prediction, reference, full_text)
    sim = semantic_similarity(prediction, reference)
    if sim > 0.75:
        return 2
    elif sim > 0.4:
        return 1
    return 0


def trim_prediction(prediction, reference, granularity):
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
    reference = example["reference"]
    result = {}

    if granularity == "partial_word":
        result["char_match"] = partial_word_char_match(prediction, reference)
    elif granularity == "next_word":
        result["first_word_match"] = first_word_match(prediction, reference)
    else:
        result["semantic_sim"] = semantic_similarity(prediction, reference)
        result["overlap"] = sequence_overlap(prediction, reference)

    trimmed = trim_prediction(prediction, reference, granularity)
    result["usefulness"] = usefulness_score(trimmed, reference, example["full_text"])
    return result


if __name__ == "__main__":
    print(first_word_match("weather today", "weather is nice"))
    print(first_word_match("climate today", "weather is nice"))
    print(partial_word_char_match("keen", "ke"))
    print(sequence_overlap("the weather is nice today", "the weather looks nice today"))
    print(sequence_overlap("the weather is nice today", "can you write me a poem"))