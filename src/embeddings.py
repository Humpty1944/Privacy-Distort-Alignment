from __future__ import annotations
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD


class WordEmbeddings:
    def __init__(self, dim: int = 32, window: int = 4, min_count: int = 2, seed: int = 0):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.vocab: List[str] = []
        self.word2idx: Dict[str, int] = {}
        self.vectors: np.ndarray | None = None  # (V, dim)

    def fit(self, text: str) -> "WordEmbeddings":
        words = text.split()
        counts = Counter(words)
        self.vocab = [w for w, c in counts.items() if c >= self.min_count]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        v = len(self.vocab)
        if v == 0:
            raise ValueError("Vocabulary is empty; lower min_count or use more text.")

        cooc = np.zeros((v, v), dtype=np.float32)
        idxs = [self.word2idx[w] for w in words if w in self.word2idx]
        for center_pos, center_idx in enumerate(idxs):
            lo = max(0, center_pos - self.window)
            hi = min(len(idxs), center_pos + self.window + 1)
            for ctx_pos in range(lo, hi):
                if ctx_pos == center_pos:
                    continue
                cooc[center_idx, idxs[ctx_pos]] += 1.0

        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True) + 1e-8
        col_sums = cooc.sum(axis=0, keepdims=True) + 1e-8
        expected = row_sums @ col_sums / max(total, 1e-8)
        pmi = np.log((cooc + 1e-8) / (expected + 1e-8))
        ppmi = np.maximum(pmi, 0.0)

        n_components = max(1, min(self.dim, v - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        vecs = svd.fit_transform(ppmi)
        if n_components < self.dim:
            vecs = np.pad(vecs, ((0, 0), (0, self.dim - n_components)))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vecs / norms
        return self

    def get(self, word: str) -> np.ndarray:
        if word in self.word2idx:
            return self.vectors[self.word2idx[word]]
        rng = np.random.default_rng(abs(hash(word)) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def nearest(self, vector: np.ndarray, exclude: str | None = None, k: int = 1) -> List[str]:
        v = vector / (np.linalg.norm(vector) + 1e-8)
        sims = self.vectors @ v
        order = np.argsort(-sims)
        out = []
        for idx in order:
            w = self.vocab[idx]
            if w == exclude:
                continue
            out.append(w)
            if len(out) >= k:
                break
        return out

    def mean_vector(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = np.stack([self.get(t) for t in tokens])
        return vecs.mean(axis=0)
