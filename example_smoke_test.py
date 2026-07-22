import random
import pandas as pd
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
from datasets import load_dataset


def _build_context(df_by_id, prompter_row, max_turns=None):
    chain = []
    current_id = prompter_row.name  
    seen = set()
    while current_id is not None and current_id in df_by_id.index:
        if current_id in seen:
            break  
        seen.add(current_id)
        row = df_by_id.loc[current_id]
        chain.append(row)
        current_id = row["parent_id"]
 
    chain.reverse()  
 
    if max_turns is not None and len(chain) > max_turns:
        chain = chain[-max_turns:]
 
    lines = []
    for row in chain:
        speaker = "User" if row["role"] == "prompter" else "Assistant"
        lines.append(f"{speaker}: {row['text']}")
 
    return "\n".join(lines)
 
 
def _load_pairs(source_splits, user_role):

    dataset_dict = load_dataset("OpenAssistant/oasst1")
 
    for s in source_splits:
        if s not in dataset_dict:
            raise ValueError(
                f"source_split '{s}' not found. "
                f"Available: {list(dataset_dict.keys())}"
            )
 
    df = pd.concat(
        [dataset_dict[s].to_pandas() for s in source_splits], ignore_index=True
    )
 
    df = df[df["lang"] == "en"]
 
    if df["message_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate message_id values found after pooling source_split(s) "
            f"{source_splits} — refusing to proceed, as this would corrupt "
            "context-chain lookups and user grouping."
        )
    df_by_id = df.set_index("message_id", drop=False)
 
    prompters = df[df["role"] == "prompter"][
        ["message_id", "text", "message_tree_id", "user_id"]
    ].rename(
        columns={
            "message_id": "parent_id",
            "text": "prompt",
            "user_id": "prompter_user_id",
        }
    )
 
    assistants = df[df["role"] == "assistant"][
        ["text", "parent_id", "message_tree_id", "user_id"]
    ].rename(columns={"user_id": "assistant_user_id"})
 
    pairs = pd.merge(
        assistants, prompters, on=["parent_id", "message_tree_id"], how="inner"
    )
    pairs["prompter_message_id"] = pairs["parent_id"]
 
    owner_col = "prompter_user_id" if user_role == "prompter" else "assistant_user_id"
    pairs["owner_user_id"] = pairs[owner_col]
    pairs = pairs.dropna(subset=["owner_user_id"])
 
    grouped = pairs.groupby("owner_user_id")
    return pairs, df_by_id, grouped
 
 
def _records_for_users(
    users, records_per_user, record_split, grouped, df_by_id, rng,
    include_context, max_context_turns,
):
    records = []
    for uid in users:
        user_pairs = grouped.get_group(uid).sample(
            n=records_per_user,
            replace=False,
            random_state=rng.randint(0, 1000000),
        )
        for _, row in user_pairs.iterrows():
            if include_context:
                prompter_row = df_by_id.loc[row["prompter_message_id"]]
                model_input = _build_context(
                    df_by_id, prompter_row, max_turns=max_context_turns
                )
            else:
                model_input = row["prompt"]
 
            records.append(
                RawRecord(
                    user_id=str(uid),
                    input=model_input,
                    target=row["text"],
                    split=record_split,
                )
            )
    return records
 
 
def make_records(
    num_users,
    records_per_user,
    source_split="train",
    record_split="train",
    user_role="prompter",
    seed=None,
    include_context=False,
    max_context_turns=None,
    exclude_users=None,
):
    source_splits = [source_split] if isinstance(source_split, str) else list(source_split)
 
    pairs, df_by_id, grouped = _load_pairs(source_splits, user_role)
 
    if exclude_users:
        pairs = pairs[~pairs["owner_user_id"].isin(exclude_users)]
        grouped = pairs.groupby("owner_user_id")
 
    eligible_users = [uid for uid, g in grouped if len(g) >= records_per_user]
 
    if len(eligible_users) < num_users:
        raise ValueError(
            f"Only {len(eligible_users)} real users in source_split(s) "
            f"{source_splits} have >= {records_per_user} records each "
            f"(after exclusions); requested {num_users} users. Try: fewer "
            "num_users, fewer records_per_user, pooling more source splits, "
            "or use make_user_disjoint_splits to partition users out of "
            "the full pooled dataset instead of relying on OASST1's "
            "predefined split sizes."
        )
 
    rng = random.Random(seed)
    chosen_users = rng.sample(eligible_users, num_users)
 
    return _records_for_users(
        chosen_users, records_per_user, record_split, grouped, df_by_id, rng,
        include_context, max_context_turns,
    )
 
 
def make_disjoint_user_splits(
    num_train_users,
    num_eval_users,
    records_per_user,
    train_source_split="train",
    eval_source_split="validation",
    user_role="prompter",
    seed=None,
    include_context=False,
    max_context_turns=None,
):

    train_records = make_records(
        num_users=num_train_users,
        records_per_user=records_per_user,
        source_split=train_source_split,
        record_split="train",
        user_role=user_role,
        seed=seed,
        include_context=include_context,
        max_context_turns=max_context_turns,
    )
    train_users = {r.user_id for r in train_records}
 
    eval_records = make_records(
        num_users=num_eval_users,
        records_per_user=records_per_user,
        source_split=eval_source_split,
        record_split="eval",
        user_role=user_role,
        seed=seed,
        include_context=include_context,
        max_context_turns=max_context_turns,
        exclude_users=train_users,
    )
    eval_users = {r.user_id for r in eval_records}
 
    assert train_users.isdisjoint(eval_users), (
        "Internal error: train/eval user sets are not disjoint."
    )
 
    return train_records, eval_records

 
def make_user_disjoint_splits(
    num_train_users,
    num_eval_users,
    train_records_per_user,
    eval_records_per_user=1,
    source_splits=("train", "validation"),
    user_role="prompter",
    seed=None,
    include_context=False,
    max_context_turns=None,
):
    pairs, df_by_id, grouped = _load_pairs(list(source_splits), user_role)

    counts = grouped.size()
 
    train_eligible = set(counts[counts >= train_records_per_user].index)
    eval_eligible = set(counts[counts >= eval_records_per_user].index)
 
    if len(train_eligible) < num_train_users:
        raise ValueError(
            f"Only {len(train_eligible)} real users across source_splits "
            f"{list(source_splits)} have >= {train_records_per_user} "
            f"records each; requested {num_train_users} train users. Try "
            "lowering train_records_per_user/num_train_users, or pooling "
            "in more source_splits."
        )
 
    rng = random.Random(seed)
 
    train_pool = list(train_eligible)
    rng.shuffle(train_pool)
    train_users = train_pool[:num_train_users]
 
    eval_pool = list(eval_eligible - set(train_users))
    if len(eval_pool) < num_eval_users:
        raise ValueError(
            f"Only {len(eval_pool)} real users (after removing train users) "
            f"have >= {eval_records_per_user} records each; requested "
            f"{num_eval_users} eval users. Try lowering "
            "eval_records_per_user/num_eval_users, or pooling in more "
            "source_splits."
        )
    rng.shuffle(eval_pool)
    eval_users = eval_pool[:num_eval_users]
 
    assert set(train_users).isdisjoint(eval_users) 
 
    train_records = _records_for_users(
        train_users, train_records_per_user, "train", grouped, df_by_id,
        rng, include_context, max_context_turns,
    )
    eval_records = _records_for_users(
        eval_users, eval_records_per_user, "eval", grouped, df_by_id,
        rng, include_context, max_context_turns,
    )
 
    return train_records, eval_records
 

config = load_config(
    ExperimentConfig,
    "configs/exp1.yaml",
    metric_computer=DefaultMetricComputer(),
    decoding=DecodingConfig(
        generation_temperatures=(0.0,),
        max_new_tokens=8,
    ),
)

torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

train_records, eval_records = make_user_disjoint_splits(
    num_train_users=536, num_eval_users=200,
    train_records_per_user=10, eval_records_per_user=1,
    seed=42,
)


all_records = train_records + eval_records 

tokenizer = load_tokenizer(config.model_name)
encoder = HFGenerationEncoder(tokenizer, config)
train_dataset, eval_dataloader = DataPreprocessor(encoder, config).build(all_records)
print(f"num_users={train_dataset.num_users}")

model_factory = make_model_factory(config=config, tokenizer=tokenizer)


decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    store = run_privacy_sweep(
        model_factory=model_factory,
        train_dataset=train_dataset,
        eval_dataloader=eval_dataloader,
        config=config,
        epsilons=[8],
        decode_fn=decode_fn,
    )

    df = store.to_dataframe()
    print("\n=== Results table ===")
    print(df)
    store.to_csv("res.csv")
