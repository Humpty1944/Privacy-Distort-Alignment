from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _isolate_nan_example(model, tokenizer, batch, step: int):
    print(f"    [NaN-CHECK] isolating step {step}'s batch example-by-example (batch_size=1 each) ...")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(batch["input_ids"].shape[0]):
            single = {k: v[i:i + 1] for k, v in batch.items()}
            try:
                out = model(**single)
                loss_val = float(out.loss.item())
            except Exception as e:  # pragma: no cover
                loss_val = float("nan")
                print(f"    [NaN-CHECK]   example {i}: raised {type(e).__name__}: {e}")
            text = tokenizer.decode(single["input_ids"][0], skip_special_tokens=True)
            n_real = int(single["attention_mask"].sum().item())
            status = "OK" if np.isfinite(loss_val) else "NON-FINITE"
            print(f"    [NaN-CHECK]   example {i} [{status}]: loss={loss_val}, "
                  f"real_tokens={n_real}, text={text[:250]!r}")
    if was_training:
        model.train()


def _check_finite(name: str, value: float, step: int, extra: str = "") -> bool:
    if not np.isfinite(value):
        print(f"    [NaN-CHECK] {name} went non-finite ({value}) at step {step}{extra} -- "
              f". Common causes: lr too high for this model/batch size, a pathological batch "
              f"(very short/empty sequence), or fp16/autocast numerical issues.")
        return False
    return True


@dataclass
class TrainLog:
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    clip_rates: List[float] = field(default_factory=list)
    noise_multiplier: float = float("nan")

    @property
    def loss_variance(self) -> float:
        return float(np.var(self.losses)) if len(self.losses) > 1 else 0.0

    @property
    def mean_grad_norm(self) -> float:
        return float(np.mean(self.grad_norms)) if self.grad_norms else float("nan")

    @property
    def mean_clip_rate(self) -> float:
        return float(np.mean(self.clip_rates)) if self.clip_rates else float("nan")


def _per_sample_grad_norms(model) -> "torch.Tensor | None":
    norms_per_param = []
    for p in model.parameters():
        gs = getattr(p, "grad_sample", None)
        if gs is not None:
            norms_per_param.append(gs.reshape(gs.shape[0], -1).norm(2, dim=1))
    if not norms_per_param:
        return None
    return torch.stack(norms_per_param, dim=0).norm(2, dim=0)


class _TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.encodings = [tokenizer(t, truncation=True, max_length=max_length)["input_ids"] for t in texts]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def _make_collate_fn(pad_token_id: int):
    def collate(batch):
        maxlen = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), maxlen), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)  # -100 = ignored by HF's loss
        attention_mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, seq in enumerate(batch):
            L = len(seq)
            input_ids[i, :L] = seq
            labels[i, :L] = seq
            attention_mask[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    return collate


def train_plain(hf_model, texts: List[str], epochs: int, lr: float,
                 batch_size: int = 8, max_grad_norm: float = 1.0,
                 warmup_frac: float = 0.1, seed: int = 0) -> TrainLog:
    torch.manual_seed(seed)
    model = hf_model.model
    model.train()
    dataset = _TextDataset(texts, hf_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_make_collate_fn(hf_model.tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = max(1, epochs * len(loader))
    num_warmup_steps = max(1, int(warmup_frac * num_training_steps))
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    log = TrainLog()
    step = 0
    broken = False
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(hf_model.device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norm_val = float(total_norm.item())
            loss_val = float(out.loss.item())
            if step <= 1:
                print(f"    [trace] step {step}: loss={loss_val}, grad_norm={grad_norm_val}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")
            if not broken:
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                ok_loss = _check_finite("loss", loss_val, step, f", batch seq_lens={seq_lens}")
                ok_grad = _check_finite("grad_norm", grad_norm_val, step)
                broken = not (ok_loss and ok_grad)
                if broken:
                    _isolate_nan_example(model, hf_model.tokenizer, batch, step)
            log.grad_norms.append(grad_norm_val)
            optimizer.step()
            scheduler.step()
            log.losses.append(loss_val)
            step += 1
            if broken:
                print(f"    [NaN-CHECK] stopping this fine-tuning run early at step {step} "
                      f"(non-finite loss/grad detected).")
                model.eval()
                return log
    model.eval()
    return log


def train_dp(hf_model, texts: List[str], epsilon: float, delta: float,
             clip_norm: float, epochs: int, lr: float, batch_size: int = 8,
             max_physical_batch_size: int = 8, seed: int = 0) -> TrainLog:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    torch.manual_seed(seed)
    model = hf_model.model
    model = ModuleValidator.fix(model)
    hf_model.model = model
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"    [warn] ModuleValidator flagged {len(errors)} potential DP-incompatibility "
              f"issue(s) in {hf_model.model_name}: {errors[:3]}{'...' if len(errors) > 3 else ''}")

    dataset = _TextDataset(texts, hf_model.tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_make_collate_fn(hf_model.tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=clip_norm,
    )
    hf_model.model = model

    noise_multiplier = float(getattr(optimizer, "noise_multiplier", float("nan")))
    log = TrainLog(noise_multiplier=noise_multiplier)
    step = 0
    broken = False
    for _ in range(epochs):
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=max_physical_batch_size,
                                 optimizer=optimizer) as memory_safe_loader:
            for batch in memory_safe_loader:
                batch = {k: v.to(hf_model.device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(**batch)
                out.loss.backward()

                per_sample_norms = _per_sample_grad_norms(model)
                loss_val = float(out.loss.item())
                if not broken:
                    seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                    ok_loss = _check_finite("loss", loss_val, step, f", batch seq_lens={seq_lens}")
                    broken = not ok_loss
                    if broken:
                        _isolate_nan_example(model, hf_model.tokenizer, batch, step)
                if per_sample_norms is not None:
                    log.grad_norms.append(float(per_sample_norms.mean().item()))
                    log.clip_rates.append(float((per_sample_norms > clip_norm).float().mean().item()))
                else:
                    print(f"    [warn] step {step}: no parameter had `.grad_sample` populated -- "
                          f"Opacus's per-sample-gradient hooks may not have attached correctly. "
                          f"mean_grad_norm/clip_rate will be NaN for this run.")

                optimizer.step()
                log.losses.append(loss_val)
                step += 1
                if broken:
                    print(f"    [NaN-CHECK] stopping this DP-SGD run early at step {step} "
                          f"(non-finite loss detected).")
                    break
        if broken:
            break

    hf_model.model = model._module if hasattr(model, "_module") else model
    hf_model.model.eval()
    spent = privacy_engine.get_epsilon(delta)
    print(f"    [dp] target eps={epsilon}, delta={delta} -> accountant reports spent eps={spent:.3f}, "
          f"noise_multiplier={log.noise_multiplier:.3f}, mean_grad_norm={log.mean_grad_norm:.3f}, "
          f"clip_rate={log.mean_clip_rate:.3f}")
    return log
