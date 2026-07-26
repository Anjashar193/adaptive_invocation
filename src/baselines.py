from collections import Counter, defaultdict
import re


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

    def predict_next_word(self, prefix):
        toks = tokenize(prefix)
        if not toks:
            return self.fallback[0] if self.fallback else "the"
        context = tuple(toks[-(self.n - 1):]) if self.n > 1 else ()
        if context in self.context_counts:
            return self.context_counts[context].most_common(1)[0][0]
        # backoff to unigram
        return self.fallback[0] if self.fallback else "the"

    def predict_top_k(self, prefix, k=5):
        toks = tokenize(prefix)
        context = tuple(toks[-(self.n - 1):]) if toks and self.n > 1 else ()
        if context in self.context_counts:
            return [w for w, _ in self.context_counts[context].most_common(k)]
        return self.fallback[:k]


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


if __name__ == "__main__":
    train_texts = [
        "can you help me write a short essay about climate change",
        "can you make a list of the best places to visit in italy",
        "what is the capital of france and why is it important",
    ]
    freq = FrequencyBaseline(); freq.fit(train_texts)
    ngram = NgramBaseline(); ngram.fit(train_texts)
    rnd = RandomBaseline(); rnd.fit(train_texts)

    print("Frequency baseline top-5:", freq.predict_top_k("can you"))
    print("Bigram baseline predict:", ngram.predict_next_word("can you"))
    print("Random baseline predict:", rnd.predict_next_word("can you"))