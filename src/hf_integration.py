"""
Wires a real Hugging Face model (classification or generation)
to work with the rest of the code
LORA OR FULL IS DECIDED HERE

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.experiment_configs import ExperimentConfig

from .data_preprocessing import FeatureEncoder
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def load_tokenizer(model_name: str):

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
        return raw_target  # eval


class HFWrapper(nn.Module):
    """
    Normalizes any HF model to the calling convention
    """

    def __init__(self, model, pad_token_id=None):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):

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


def make_model_factory(
    config: ExperimentConfig,
    tokenizer,
    num_classes=None,
):

    def _factory():
        torch.manual_seed(config.seed)
        if config.task == "classification":
            base = AutoModelForSequenceClassification.from_pretrained(
                config.model_name, num_labels=num_classes
            )
        elif config.task == "generation":
            base = AutoModelForCausalLM.from_pretrained(config.model_name)
        else:
            raise ValueError(f"Unknown task: {config.task}")
        base.config.pad_token_id = tokenizer.pad_token_id

        if config.lora["enabled"]:

            lora_config = LoraConfig(
                r=config.lora["r"],
                lora_alpha=config.lora["alpha"],
                target_modules=config.lora["target_modules"],
                lora_dropout=config.lora["dropout"],
                bias=config.lora["bias"],
                task_type="CAUSAL_LM" if config.task == "generation" else "SEQ_CLS",
            )
            base = get_peft_model(base, lora_config)
        return HFWrapper(base, pad_token_id=tokenizer.pad_token_id)

    return _factory
