"""
Wires a real Hugging Face model (classification or causal-LM generation)
into the pluggable boundaries this package already exposes:

    FeatureEncoder -> HFClassificationEncoder / HFGenerationEncoder
    model_factory -> make_model_factory(...)
    collator   -> hf_collator

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data_preprocessing import FeatureEncoder


def load_tokenizer(model_name: str):
    """
    Load a tokenizer and make sure it has a pad token.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@dataclass
class HFEncoderConfig:
    max_length: int = 256


class HFClassificationEncoder(FeatureEncoder):
    def __init__(self, tokenizer, config: HFEncoderConfig = HFEncoderConfig()):
        self.tokenizer = tokenizer
        self.config = config

    def encode_input(self, raw_input):
        enc = self.tokenizer(
            raw_input,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def encode_target(self, raw_target, split):
        return torch.as_tensor(raw_target, dtype=torch.long)


class HFGenerationEncoder(FeatureEncoder):

    def __init__(self, tokenizer, config: HFEncoderConfig = HFEncoderConfig()):
        self.tokenizer = tokenizer
        self.config = config

    def encode_input(self, raw_input):
        prior_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                raw_input,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = prior_side
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def encode_target(self, raw_target, split):
        if split == "train":
            enc = self.tokenizer(
                raw_target,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            labels = enc["input_ids"][0].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return labels
        return raw_target  # eval: keep raw reference text for BLEU/ROUGE


def hf_collator(records):
    """Assembles the dict-shaped {input_ids, attention_mask, labels} batch
    that HF models expect, from encode_input_dict, encode_target"""
    input_ids = torch.stack([r[0]["input_ids"] for r in records])
    attention_mask = torch.stack([r[0]["attention_mask"] for r in records])
    label_items = [r[1] for r in records]
    labels = (
        torch.stack(label_items)
        if torch.is_tensor(label_items[0])
        else torch.as_tensor(label_items)
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class HFWrapper(nn.Module):
    """
    Normalizes any HF model to the calling convention the rest of this
    package expects:
      - training path (labels given via **batch)         -> full HF output (has .loss)
      - eval/classification path (no labels, no cache)    -> raw logits tensor
      - cache-aware generation path (use_cache=True)       -> full HF output (has .past_key_values)
    """

    def __init__(self, model, pad_token_id=None):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False):
        if attention_mask is None and self.pad_token_id is not None and past_key_values is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if labels is not None or use_cache:
            return outputs
        return outputs.logits


def make_model_factory(model_name: str, task: str, tokenizer, num_classes: int = None, seed: int = 42):
    """
    Returns a zero-arg callable suitable for `run_experiment`/`run_privacy_sweep`'s
    `model_factory` argument -- NOT this function itself. Call it once and pass
    the *result* as `model_factory=...`:

        factory = make_model_factory(...)
        run_privacy_sweep(model_factory=factory, ...)
    """
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    def _factory():
        torch.manual_seed(seed)
        if task == "classification":
            base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        elif task == "generation":
            base = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown task: {task}")
        base.config.pad_token_id = tokenizer.pad_token_id
        return HFWrapper(base, pad_token_id=tokenizer.pad_token_id)

    return _factory
