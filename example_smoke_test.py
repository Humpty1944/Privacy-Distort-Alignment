"""
Smoke test: dp_finetune against a real, tiny Hugging Face model
(sshleifer/tiny-gpt2 -- a randomly-initialized GPT-2 config kept on the
Hub specifically for fast tests like this one).

Requires network access to huggingface.co. If that's unavailable, this
will fail at the `from_pretrained` calls with a connection error -- the
rest of the script (data, encoder, config) doesn't depend on the network
and can be sanity-checked independently.
"""

import random

import torch

from dp_finetune import (
    DataPreprocessor,
    DecodingConfig,
    ExperimentConfig,
    HFGenerationEncoder,
    RawRecord,
    hf_collator,
    load_tokenizer,
    make_model_factory,
    run_privacy_sweep,
)
from dp_finetune.config_load import load_config
from dp_finetune.metrics import DefaultMetricComputer

torch.manual_seed(0)
random.seed(0)

MODEL_NAME = "sshleifer/tiny-gpt2"
MAX_LENGTH = 16
NUM_USERS = 10000
RECORDS_PER_USER = 8


tokenizer = load_tokenizer(MODEL_NAME)


PROMPTS_AND_CONTINUATIONS = [
    ("the weather today is", "sunny and warm"),
    ("my favorite food is", "pizza with cheese"),
    ("the movie was", "really good tonight"),
    ("i went to the", "store to buy milk"),
]


def make_records(num_users, records_per_user, split):
    records = []
    for i in range(num_users):
        for _ in range(records_per_user):
            prompt, continuation = random.choice(PROMPTS_AND_CONTINUATIONS)
            records.append(
                RawRecord(user_id=f"user_{i}", input=prompt, target=continuation, split=split)
            )
    return records


raw_records = make_records(NUM_USERS, RECORDS_PER_USER, split="train") + make_records(
    num_users=5, records_per_user=2, split="eval"
)
config = load_config(
    ExperimentConfig,
    "configs/exp1.yaml",
    task="generation",
    metric_computer=DefaultMetricComputer(),
    decoding=DecodingConfig(
        generation_temperatures=(0.0,),  # reduced grid for the smoke test
        max_new_tokens=8,
    ),
)

encoder = HFGenerationEncoder(tokenizer, config)
train_dataset, eval_dataloader = DataPreprocessor(encoder).build(raw_records)
print(f"num_users={train_dataset.num_users}")

model_factory = make_model_factory(MODEL_NAME, task="generation", tokenizer=tokenizer)


decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    store = run_privacy_sweep(
        model_factory=model_factory,
        train_dataset=train_dataset,
        eval_dataloader=eval_dataloader,
        config=config,
        epsilons=[float("inf"), 4],  # reduced grid for the smoke test
        decode_fn=decode_fn,
    )

    df = store.to_dataframe()
    print("\n=== Results table ===")
    print(df[["training_type", "epsilon", "sigma", "bleu", "rouge1", "runtime_seconds"]])
    print("\nHF smoke test (sshleifer/tiny-gpt2) PASSED")
    store.to_csv("res.csv")
