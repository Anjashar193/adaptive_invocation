from collections import Counter, defaultdict
import re

PHRASE_WORD_COUNT = 5
SENTENCE_WORD_COUNT = 15


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


class FrequencyBaseline:
    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(tokenize(t))
        self.top_words = [w for w, _ in counts.most_common(20)]

    def predict_next_word(self, prefix):
        return self.top_words[0] if self.top_words else "the"

    def predict_top_k(self, prefix, k=5):
        return self.top_words[:k]

    def predict_top_k_chain(self, prefix, granularity, k=5):
        length = PHRASE_WORD_COUNT if granularity == "phrase" else SENTENCE_WORD_COUNT
        if not self.top_words:
            return [" ".join(["the"] * length) for _ in range(k)]
        chains = []
        for start in range(min(k, len(self.top_words))):
            words = [self.top_words[start]]
            pool = [w for i, w in enumerate(self.top_words) if i != start]
            words += pool[: length - 1]
            chains.append(" ".join(words))
        return chains


class NgramBaseline:
    def fit(self, texts, n=2):
        self.n = n
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        for t in texts:
            toks = tokenize(t)
            self.unigram_counts.update(toks)
            for i in range(len(toks) - 1):
                context = tuple(toks[max(0, i - (n - 1)):i + 1])
                self.context_counts[context].update([toks[i + 1]])
        self.fallback = [w for w, _ in self.unigram_counts.most_common(20)]
        import random as _random
        self._rng = _random.Random(42)  # fixed seed for reproducible chains

    def predict_next_word(self, prefix):
        toks = tokenize(prefix)
        if not toks:
            return self.fallback[0] if self.fallback else "the"
        context = tuple(toks[-(self.n - 1):]) if self.n > 1 else ()
        if context in self.context_counts:
            return self.context_counts[context].most_common(1)[0][0]
        return self.fallback[0] if self.fallback else "the"

    def predict_top_k(self, prefix, k=5):
        toks = tokenize(prefix)
        context = tuple(toks[-(self.n - 1):]) if toks and self.n > 1 else ()
        if context in self.context_counts:
            return [w for w, _ in self.context_counts[context].most_common(k)]
        return self.fallback[:k]

    def _generate_chain(self, start_context_toks, first_word, length, rng):
        words = [first_word]
        context_toks = list(start_context_toks) + [first_word]
        for _ in range(length - 1):
            context = tuple(context_toks[-(self.n - 1):]) if self.n > 1 else ()
            candidates, weights = [], []

            if context in self.context_counts:
                for word, count in self.context_counts[context].most_common(10):
                    candidates.append(word)
                    weights.append(count)

            if not candidates and self.fallback:
                candidates = self.fallback[:10]
                weights = [1] * len(candidates)

            next_word = rng.choices(candidates, weights=weights, k=1)[0] if candidates else "the"
            words.append(next_word)
            context_toks.append(next_word)
        return " ".join(words)

    def predict_top_k_chain(self, prefix, granularity, k=5):
        length = PHRASE_WORD_COUNT if granularity == "phrase" else SENTENCE_WORD_COUNT
        toks = tokenize(prefix)
        first_word_candidates = self.predict_top_k(prefix, k=k)
        if not first_word_candidates:
            first_word_candidates = ["the"]
        return [self._generate_chain(toks, w, length, self._rng) for w in first_word_candidates]


class RandomBaseline:
    def fit(self, texts):
        import random as _random
        self._random = _random
        vocab = set()
        for t in texts:
            vocab.update(tokenize(t))
        self.vocab = list(vocab)

    def predict_next_word(self, prefix):
        return self._random.choice(self.vocab) if self.vocab else "the"

    def predict_top_k(self, prefix, k=5):
        return self._random.sample(self.vocab, min(k, len(self.vocab))) if self.vocab else ["the"]

    def predict_top_k_chain(self, prefix, granularity, k=5):
        length = PHRASE_WORD_COUNT if granularity == "phrase" else SENTENCE_WORD_COUNT
        if not self.vocab:
            return [" ".join(["the"] * length) for _ in range(k)]
        chains = []
        for _ in range(k):
            words = self._random.choices(self.vocab, k=length)
            chains.append(" ".join(words))
        return chains


if __name__ == "__main__":
    train_texts = [
        "can you help me write a short essay about climate change",
        "can you make a list of the best places to visit in italy",
        "what is the capital of france and why is it important",
        "please explain how photosynthesis works in simple terms",
    ]
    freq = FrequencyBaseline(); freq.fit(train_texts)
    ngram = NgramBaseline(); ngram.fit(train_texts)
    rnd = RandomBaseline(); rnd.fit(train_texts)

    print("Frequency phrase chain:", freq.predict_top_k_chain("can you", "phrase", k=3))
    print("Ngram phrase chain:", ngram.predict_top_k_chain("can you", "phrase", k=3))
    print("Random phrase chain:", rnd.predict_top_k_chain("can you", "phrase", k=3))