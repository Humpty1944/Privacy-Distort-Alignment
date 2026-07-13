from __future__ import annotations
from typing import List

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import WordEmbeddings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def input_text_distortion(original: str, privatized: str, embeddings: WordEmbeddings,
                           sentence_encoder=None) -> dict:
    orig_tokens = original.split(" ")
    priv_tokens = privatized.split(" ")

    if sentence_encoder is not None:
        orig_vec = sentence_encoder.encode(original)
        priv_vec = sentence_encoder.encode(privatized)
    else:
        orig_vec = embeddings.mean_vector(orig_tokens)
        priv_vec = embeddings.mean_vector(priv_tokens)
    embedding_similarity = cosine_sim(orig_vec, priv_vec)

    informative = [t for t in orig_tokens if len(t) > 3]
    if informative:
        preserved = sum(1 for t in informative if t in priv_tokens) / len(informative)
    else:
        preserved = 1.0

    distortion = float(np.clip(1.0 - embedding_similarity, 0.0, 2.0))
    return {
        "embedding_similarity": embedding_similarity,
        "content_preservation_rate": preserved,
        "input_text_distortion": distortion,
    }


def representation_distortion(baseline_predict_fn, condition_predict_fn,
                               probe_prompts: List[str]) -> float:
    divs = []
    for prompt in probe_prompts:
        p = baseline_predict_fn(prompt)
        q = condition_predict_fn(prompt)
        divs.append(float(jensenshannon(p, q, base=2)))
    return float(np.mean(divs)) if divs else 0.0
