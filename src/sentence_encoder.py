from __future__ import annotations
from typing import Optional

import numpy as np

DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"


class SentenceEncoder:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, convert_to_numpy=True))


def get_sentence_encoder(model_name: str = DEFAULT_SENTENCE_MODEL) -> Optional[SentenceEncoder]:
    try:
        return SentenceEncoder(model_name)
    except ImportError:
        print("    [warn] sentence-transformers not installed -- falling back to "
              "mean-pooled word embeddings for embedding_similarity. "
              "`pip install sentence-transformers` for the real thing.")
        return None
