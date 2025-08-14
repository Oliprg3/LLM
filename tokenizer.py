import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merge_rules = {}

    def build_vocab(self, texts):
        words = re.findall(r"\S+|\n", texts)
        freqs = Counter(words)
        vocab = {tuple(word): freq for word, freq in freqs.items()}
        while len(vocab) < self.vocab_size:
            pairs = Counter()
            for word, freq in vocab.items():
                for i in range(len(word)-1):
                    pairs[(word[i], word[i+1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merge_rules[best] = len(self.merge_rules)
            vocab = self.merge_vocab(vocab, best)
        self.vocab = {k:"".join(k) for k in vocab.keys()}
        self.stoi = {v:k for k,v in enumerate(self.vocab.values())}
        self.itos = {i:v for v,i in self.stoi.items()}

    def merge_vocab(self, vocab, pair):
        new_vocab = {}
        for word,freq in vocab.items():
            w = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == pair:
                    w.append(word[i]+word[i+1])
                    i+=2
                else:
                    w.append(word[i])
                    i+=1
            new_vocab[tuple(w)] = freq
        return new_vocab

    def encode(self, text):
        tokens = re.findall(r"\S+|\n", text)
        indices = [self.stoi[t] for t in tokens if t in self.stoi]
        return indices

    def decode(self, indices):
        return "".join([self.itos[i] for i in indices])
