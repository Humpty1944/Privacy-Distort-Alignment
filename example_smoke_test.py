import random

import torch

from src import (
    DataPreprocessor,
    DecodingConfig,
    ExperimentConfig,
    HFGenerationEncoder,
    RawRecord,
    load_tokenizer,
    make_model_factory,
    run_privacy_sweep,
)
from src.config_load import load_config
from src.metrics import DefaultMetricComputer

MAX_LENGTH = 16
NUM_USERS = 10000
RECORDS_PER_USER = 8


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
                RawRecord(
                    user_id=f"user_{i}", input=prompt, target=continuation, split=split
                )
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
        generation_temperatures=(0.0,),
        max_new_tokens=8,
    ),
)

torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

tokenizer = load_tokenizer(config.model_name)
encoder = HFGenerationEncoder(tokenizer, config)
train_dataset, eval_dataloader = DataPreprocessor(encoder, config).build(raw_records)
print(f"num_users={train_dataset.num_users}")

model_factory = make_model_factory(config=config, tokenizer=tokenizer)


decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    store = run_privacy_sweep(
        model_factory=model_factory,
        train_dataset=train_dataset,
        eval_dataloader=eval_dataloader,
        config=config,
        epsilons=[4],
        decode_fn=decode_fn,
    )

    df = store.to_dataframe()
    print("\n=== Results table ===")
    print(
        df[["training_type", "epsilon", "sigma", "bleu", "rouge1", "runtime_seconds"]]
    )
    store.to_csv("res.csv")
