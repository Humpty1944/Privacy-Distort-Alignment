import json
from pathlib import Path


def normalize(text: str) -> str:
    return text.strip().lower()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(data, file_path):

    ensure_dir(Path(file_path).parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)