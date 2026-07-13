from __future__ import annotations
from typing import List

import numpy as np

from .embeddings import WordEmbeddings


class Mechanism:
    name: str = "base"

    def privatize(self, text: str, epsilon: float, rng: np.random.Generator) -> str:
        raise NotImplementedError


class NoPrivacy(Mechanism):
    name = "no_privacy"

    def privatize(self, text: str, epsilon: float, rng: np.random.Generator) -> str:
        return text


class RandomTokenPerturbation(Mechanism):
    name = "random_token_perturbation"

    def __init__(self, vocab: List[str]):
        self.vocab = vocab

    def privatize(self, text: str, epsilon: float, rng: np.random.Generator) -> str:
        p = float(np.clip(np.exp(-epsilon), 0.0, 1.0))
        tokens = text.split(" ")
        out = []
        for tok in tokens:
            r = rng.random()
            if r < p / 2:
                continue  # delete
            elif r < p:
                out.append(rng.choice(self.vocab) if self.vocab else tok)
            else:
                out.append(tok)
        return " ".join(out) if out else text


class SemanticSanitization(Mechanism):
    name = "semantic_sanitization"

    def __init__(self, embeddings: WordEmbeddings, noise_scale: float = 1.0):
        self.embeddings = embeddings
        self.noise_scale = noise_scale

    def privatize(self, text: str, epsilon: float, rng: np.random.Generator) -> str:
        tokens = text.split(" ")
        out = []
        scale = self.noise_scale / max(epsilon, 1e-3)
        for tok in tokens:
            vec = self.embeddings.get(tok)
            noise = rng.laplace(loc=0.0, scale=scale, size=vec.shape)
            perturbed = vec + noise
            candidates = self.embeddings.nearest(perturbed, k=1)
            out.append(candidates[0] if candidates else tok)
        return " ".join(out)


def get_mechanism(name: str, vocab: List[str], embeddings: WordEmbeddings,
                   semantic_noise_scale: float = 1.0) -> Mechanism:
    if name == "no_privacy":
        return NoPrivacy()
    if name == "random_token_perturbation":
        return RandomTokenPerturbation(vocab)
    if name == "semantic_sanitization":
        return SemanticSanitization(embeddings, noise_scale=semantic_noise_scale)
    raise ValueError(f"Unknown mechanism: {name}")
