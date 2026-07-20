"""
Decoder evaluation

For classification:
    temperature = 0           

For generation:
    temperature = 0              
    temperature = 0.7
    temperature = 1.0

Kept model-agnostic: model only needs to expose
    logits = model(input_ids)                     # classification, shape [B, num_classes]
or
    logits = model(input_ids)                      # generation, shape [B, T, vocab]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from dp_finetune.experiment_configs import DecodingConfig



def _call_model_eval(model, x, device: str, **kwargs):
    """
    Calls the model for eval/generation and returns a raw logits
    tensor.

    x may be:
      - a plain tensor (input_ids only)             -> model(x)
      - a dict, e.g. {"input_ids":..., "attention_mask":...} -> model(**x)
    """
    if isinstance(x, dict):
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        out = model(**x, **kwargs)
    else:
        x = x.to(device)
        out = model(x, **kwargs)

    if kwargs.get("use_cache"):
        return out  # caller needs .logits and .past_key_values
    return out.logits if hasattr(out, "logits") else out


def sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: [B, vocab]. temperature == 0 -> argmax.
    Otherwise scale logits by 1/temperature and sample from the softmax.
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _flatten_eval_labels(labels) -> List[int]:
    """
    Normalize label batches from the eval dataloader into a flat Python list.
    Some encoders may yield a tensor, some may yield a tuple/list of scalar
    labels, and some may yield a single scalar item that needs to be wrapped.
    """
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().tolist()

    if isinstance(labels, (list, tuple)):
        flat: List[int] = []
        for item in labels:
            if isinstance(item, torch.Tensor):
                flat.extend(item.detach().cpu().tolist())
            else:
                flat.append(item)
        return flat

    return [labels]


def evaluate_classification(
    model: torch.nn.Module,
    dataloader,
    device: str = "cpu",
    temperature: float = 0.0,
) -> dict:
    """
    Runs the classifier with temperature=0 and
    returns raw predictions/labels
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in dataloader:
            logits = _call_model_eval(model, x, device)
            preds = sample_next_token(logits, temperature=temperature)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(_flatten_eval_labels(y))
    return {"predictions": all_preds, "labels": all_labels}


def generate(
    model: torch.nn.Module,
    input_ids,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Greedy or temperature decoding; the last position's logits are used to
    pick the next token at each step.

    `input_ids` may be a plain tensor, or a dict with "input_ids" and
    (optionally) "attention_mask".

    Reuses past_key_values (KV-cache) across steps when the model supports
    it, so each step only forwards the newest token instead of recomputing
    the whole growing sequence.
    """
    model.eval()
    if isinstance(input_ids, dict):
        ids = input_ids["input_ids"].to(device)
        attention_mask = input_ids.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
    else:
        ids = input_ids.to(device)
        attention_mask = None

    def _full_recompute():
        if attention_mask is not None:
            out = model(input_ids=ids, attention_mask=attention_mask)
        else:
            out = model(ids)
        return out.logits if hasattr(out, "logits") else out

    past_key_values = None
    use_cache = True  # optimistic; disabled permanently on first incompatibility

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = None
            if use_cache:
                step_ids = ids[:, -1:] if past_key_values is not None else ids
                try:
                    out = model(
                        input_ids=step_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                except TypeError:
                    use_cache, past_key_values = False, None
                else:
                    if hasattr(out, "logits") and getattr(out, "past_key_values", None) is not None:
                        logits, past_key_values = out.logits, out.past_key_values
                    else:
                        use_cache, past_key_values = False, None

            if logits is None:
                logits = _full_recompute()

            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(next_token_logits, temperature=temperature)
            ids = torch.cat([ids, next_token.unsqueeze(-1)], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token).unsqueeze(-1)], dim=-1
                )
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
    return ids


def evaluate_generation(
    model: torch.nn.Module,
    dataloader,
    config: DecodingConfig,
    device: str = "cpu",
    decode_fn: Optional[Callable[[torch.Tensor], str]] = None,
) -> dict:
    """
    Runs generation and returns hypotheses 
    and references, if the dataloader yields them
    keyed by temperature
    """
    results = {}
    for temperature in config.generation_temperatures:
        hyps: List = []
        refs: List = []
        for batch in dataloader:
            prompt_ids, reference = batch
            out_ids = generate(
                model,
                prompt_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=temperature,
                eos_token_id=config.eos_token_id,
                device=device,
            )
            if decode_fn is not None:
                hyps.extend(decode_fn(o) for o in out_ids)
                refs.extend(decode_fn(r) if torch.is_tensor(r) else r for r in reference)
            else:
                hyps.extend(out_ids.cpu().tolist())
                refs.extend(reference)
        results[temperature] = {"hypotheses": hyps, "references": refs}
    return results
