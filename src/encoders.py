import torch

from dp_finetune.data_preprocessing import FeatureEncoder
from dp_finetune.experiment_configs import ExperimentConfig


class HFClassificationEncoder(FeatureEncoder):
    def __init__(self, tokenizer, config: ExperimentConfig):
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
    def __init__(self, tokenizer, config: ExperimentConfig):
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
        if split == "train":
            enc = self.tokenizer(raw_target, truncation=True, padding="max_length",
                                 max_length=self.config.max_length, return_tensors="pt")
            labels = enc["input_ids"][0].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return labels
        return raw_target