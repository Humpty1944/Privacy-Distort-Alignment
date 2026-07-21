from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationConfig:
    output_dir: Path = Path("outputs")

    truthfulqa_max_new_tokens: int = 64
    ifeval_max_new_tokens: int = 128
    alpaca_max_new_tokens: int = 128

    batch_size: int = 1

    save_json: bool = True
    save_csv: bool = True

    deterministic: bool = True