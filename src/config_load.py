# config_loader.py
import json
from dataclasses import fields, is_dataclass
import yaml  

def load_config(cls, path: str, **overrides):
    """
    Load a YAML/JSON file into a dataclass instance.
    Unknown keys in the file raise loudly instead of being silently
    ignored
    """
    with open(path) as f:
        data = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)

    data.update(overrides)  

    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")

    for f in fields(cls):
        if is_dataclass(f.type) and isinstance(data.get(f.name), dict):
            data[f.name] = f.type(**data[f.name])

    return cls(**data)