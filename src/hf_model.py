from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "src.hf_model requires torch + transformers. Install them "
        "(see requirements.txt / COLAB.md) on your cloud instance."
    ) from e


DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m"



@dataclass
class HFModel:
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32).to(self.device)
        actual_dtype = next(self.model.parameters()).dtype
        print(f"    [dtype-check] {self.model_name} loaded with dtype={actual_dtype} on {self.device} "
              f"(pad_token_id={self.tokenizer.pad_token_id}) -- should be torch.float32.")

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    def reset_from(self, other: "HFModel"):
        self.model.load_state_dict(other.model.state_dict())

    @torch.no_grad()
    def next_token_probs(self, context: str) -> np.ndarray:
        self.model.eval()
        enc = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**enc).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        return probs

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24) -> str:
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_fn(self):
        return lambda prompt, max_new_tokens=24: self.generate(prompt, max_new_tokens)

    def predict_fn(self):
        return lambda prompt: self.next_token_probs(prompt)

    def calibration_ece(self, probes, n_bins: int = 5) -> float:
        self.model.eval()
        confidences, corrects = [], []
        with torch.no_grad():
            for p in probes:
                prompt_ids = self.tokenizer(p.prompt, return_tensors="pt").input_ids[0]
                full_ids = self.tokenizer(f"{p.prompt} {p.target}", return_tensors="pt").input_ids[0]
                if len(full_ids) <= len(prompt_ids):
                    continue
                full_ids = full_ids.to(self.device).unsqueeze(0)
                logits = self.model(input_ids=full_ids).logits[0]
                probs = torch.softmax(logits, dim=-1)
                for t in range(len(prompt_ids) - 1, len(full_ids[0]) - 1):
                    pred = int(torch.argmax(probs[t]).item())
                    confidences.append(float(probs[t, pred].item()))
                    corrects.append(1.0 if pred == int(full_ids[0, t + 1].item()) else 0.0)
        if not confidences:
            return float("nan")
        confidences = np.array(confidences)
        corrects = np.array(corrects)
        bins = np.linspace(0, 1, n_bins + 1)
        ece, n = 0.0, len(confidences)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(confidences[mask].mean() - corrects[mask].mean())
        return float(ece)
