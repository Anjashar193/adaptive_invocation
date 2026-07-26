
import re
import random
from datasets import load_dataset

random.seed(42)

SPLIT_RATIOS = [0.15, 0.3, 0.5, 0.7, 0.85]  


def load_oasst2_prompter_en(split="validation", max_messages=500):
    ds = load_dataset("OpenAssistant/oasst2", split=split)
    filtered = [
        row["text"] for row in ds
        if row.get("role") == "prompter" and row.get("lang") == "en"
        and row["text"] and len(row["text"].split()) >= 6  
    ]
    random.shuffle(filtered)
    return filtered[:max_messages]


def word_split(text, ratio):
    words = text.split()
    cut = max(1, int(len(words) * ratio))
    cut = min(cut, len(words) - 1)  
    prefix = " ".join(words[:cut])
    continuation = " ".join(words[cut:])
    return prefix, continuation


def make_partial_word_example(text, ratio):
    words = text.split()
    cut = max(1, int(len(words) * ratio))
    cut = min(cut, len(words) - 1)
    target_word = words[cut]
    if len(target_word) < 3:
        return None  
    split_point = random.randint(1, len(target_word) - 1)
    partial = target_word[:split_point]
    remainder = target_word[split_point:]
    prefix = " ".join(words[:cut] + [partial])
    return prefix, remainder, target_word


def build_eval_set(messages, granularity):
    examples = []
    for text in messages:
        for ratio in SPLIT_RATIOS:
            if granularity == "partial_word":
                result = make_partial_word_example(text, ratio)
                if result is None:
                    continue
                prefix, remainder, full_word = result
                examples.append({"prefix": prefix, "reference": remainder,
                                  "full_word": full_word, "full_text": text})

            elif granularity == "next_word":
                prefix, continuation = word_split(text, ratio)
                cont_words = continuation.split()
                if not cont_words:
                    continue
                examples.append({"prefix": prefix, "reference": cont_words[0], "full_text": text})

            elif granularity == "phrase":
                prefix, continuation = word_split(text, ratio)
                phrase = " ".join(continuation.split()[:5])
                if not phrase:
                    continue
                examples.append({"prefix": prefix, "reference": phrase, "full_text": text})

            elif granularity == "sentence":
                prefix, continuation = word_split(text, ratio)
                sentence_match = re.split(r'(?<=[.!?]) ', continuation)
                if not sentence_match or not sentence_match[0]:
                    continue
                examples.append({"prefix": prefix, "reference": sentence_match[0], "full_text": text})

    return examples


if __name__ == "__main__":
    messages = load_oasst2_prompter_en(max_messages=200)
    print(f"Loaded {len(messages)} prompter messages")
    for g in ["partial_word", "next_word", "phrase", "sentence"]:
        ex = build_eval_set(messages, g)
        print(f"{g}: {len(ex)} examples")