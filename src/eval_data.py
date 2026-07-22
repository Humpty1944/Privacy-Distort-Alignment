"""
Evaluation

For classification:
    temperature = 0

For generation:
    temperature = 0
    temperature = 0.7
    temperature = 1.0

    logits = model(input_ids) -> classification, shape [B, num_classes]
or
    logits = model(input_ids) -> generation, shape [B, T, vocab]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiment_configs import DecodingConfig


def _call_model_eval(model, x, device, **kwargs):
    if isinstance(x, dict):
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        out = model(**x, **kwargs)
    else:
        x = x.to(device)
        out = model(x, **kwargs)

    if kwargs.get("use_cache"):
        return out
    return out.logits if hasattr(out, "logits") else out


def sample_from_logits(logits, temperature):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _flatten_eval_labels(labels):
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().tolist()

    if isinstance(labels, (list, tuple)):
        flat = []
        for item in labels:
            if isinstance(item, torch.Tensor):
                flat.extend(item.detach().cpu().tolist())
            else:
                flat.append(item)
        return flat

    return [labels]


def evaluate_classification(
    model,
    dataloader,
    device="cpu",
    temperature=0.0,
):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            logits = _call_model_eval(model, x, device)
            preds = sample_from_logits(logits, temperature=temperature)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(_flatten_eval_labels(y))
    return {"predictions": all_preds, "labels": all_labels}


def _full_recompute(model, ids, attention_mask):
    if attention_mask is not None:
        out = model(input_ids=ids, attention_mask=attention_mask)
    else:
        out = model(ids)
    return out.logits if hasattr(out, "logits") else out


def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature,
    eos_token_id=None,
    device="cpu",
):

    model.eval()
    if isinstance(input_ids, dict):
        ids = input_ids["input_ids"].to(device)
        attention_mask = input_ids.get("attention_mask")
        attention_mask = (
            attention_mask.to(device) if attention_mask is not None else None
        )
    else:
        ids = input_ids.to(device)
        attention_mask = None

    past_key_values = None
    use_cache = True

    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if finished.all():
                break
            logits = None
            if use_cache:
                step_ids = ids[:, -1:] if past_key_values is not None else ids
                step_x = {"input_ids": step_ids, "attention_mask": attention_mask}
                try:
                    out = _call_model_eval(
                        model,
                        step_x,
                        device,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                except TypeError:
                    use_cache, past_key_values = False, None
                else:
                    if (
                        hasattr(out, "logits")
                        and getattr(out, "past_key_values", None) is not None
                    ):
                        logits, past_key_values = out.logits, out.past_key_values
                    else:
                        use_cache, past_key_values = False, None

            if logits is None:
                logits = _full_recompute(model, ids, attention_mask)

            next_token_logits = logits[:, -1, :]
            next_token = sample_from_logits(next_token_logits, temperature=temperature)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
            ids = torch.cat([ids, next_token.unsqueeze(-1)], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token).unsqueeze(-1)], dim=-1
                )
            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
    return ids


def _truncate_at_eos(token_ids, eos_token_id):
    if eos_token_id is None:
        return token_ids
    try:
        eos_pos = token_ids.index(eos_token_id)
        return token_ids[:eos_pos]
    except ValueError:
        return token_ids


def evaluate_generation(
    model,
    dataloader,
    config: DecodingConfig,
    device="cpu",
    decode_fn=None,
):
    results = {}
    for temperature in config.generation_temperatures:
        hyps = []
        refs = []
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
            for o in out_ids:
                token_list = _truncate_at_eos(o.cpu().tolist(), config.eos_token_id)
                if decode_fn is not None:
                    hyps.append(decode_fn(torch.tensor(token_list)))
                else:
                    hyps.append(token_list)

            if decode_fn is not None:
                refs.extend(
                    decode_fn(r) if torch.is_tensor(r) else r for r in reference
                )
            else:
                refs.extend(reference)
        results[temperature] = {"hypotheses": hyps, "references": refs}
    return results
