from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from src.training_configs import DPConfig

from .privacy_accountant import PrivacyAccountant, PrivacyReport
from .user_level_dataset import UserLevelDataset


@dataclass
class DPResult:
    privacy_report: PrivacyReport
    loss_curve: List[float]
    grad_norm_curve: List[float]
    epsilon_curve: List[float] = field(default_factory=list)
    sigma: Optional[float] = None
    clipped_rate: float = 0


def _resolve_autocast_dtype(cfg: DPConfig) -> Optional[torch.dtype]:
    """
    Map a config-level string to a torch dtype for autocast
    """
    dtype_name = cfg.mixed_precision
    if dtype_name is None:
        return None
    dtype_name = dtype_name.lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    return None


def _stack_collated_values(values):
    """Stack a list of tensors or scalar values into a single batch tensor"""
    if not values:
        raise ValueError("Cannot collate an empty record set.")

    if all(isinstance(v, torch.Tensor) for v in values):
        return torch.stack(values)

    if all(isinstance(v, (int, float)) for v in values):
        return torch.as_tensor(values)

    return torch.as_tensor(values)


def _stack_collated_inputs(inputs):
    """Stack either a list of tensors or a list of dicts keyed by tensor fields"""
    if not inputs:
        raise ValueError("Cannot collate an empty record set.")

    first = inputs[0]
    if isinstance(first, dict):
        stacked = {}
        for key in first:
            values = [item[key] for item in inputs]
            stacked[key] = _stack_collated_values(values)
        return stacked

    return _stack_collated_values(inputs)


def _default_hf_user_collator(records):
    """Collapse a list of per-user records into one batch"""
    inputs = []
    labels = []
    for item in records:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            raise ValueError(
                "_default_hf_user_collator expects each record to be (input, label)."
            )
        inputs.append(x)
        labels.append(y)

    stacked_inputs = _stack_collated_inputs(inputs)
    stacked_labels = _stack_collated_values(labels)
    if isinstance(stacked_inputs, dict):
        return {**stacked_inputs, "labels": stacked_labels}
    return {"input_ids": stacked_inputs, "labels": stacked_labels}


def _call_model_with_batch(model, batch: dict, device: str):
    try:
        return model(**batch)
    except TypeError:
        input_tensor = batch.get("input_ids")
        if input_tensor is None:
            tensor_values = [v for v in batch.values() if isinstance(v, torch.Tensor)]
            if not tensor_values:
                raise
            input_tensor = tensor_values[0]
        return model(input_tensor)


def _compute_user_gradient_batched(
    model,
    collator: Callable,
    records,
    device,
    autocast_dtype,
    loss_fn=None,
):
    """
    g_{t,i}: one user's average per-user gradient, computed with a single
    forward/backward pass over the user's whole record set.
    """
    model.zero_grad(set_to_none=True)

    batch = collator(records)
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    device_type = "cuda" if "cuda" in device else "cpu"

    if autocast_dtype is not None:
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            outputs = _call_model_with_batch(model, batch, device)
            loss = outputs.loss if hasattr(outputs, "loss") else None
    else:
        outputs = _call_model_with_batch(model, batch, device)
        loss = outputs.loss if hasattr(outputs, "loss") else None

    if loss is None:
        if loss_fn is None:
            raise ValueError(
                "A loss function is required when the model does not expose outputs.loss."
            )
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        labels = batch.get("labels")
        if labels is None:
            raise ValueError(
                "labels must be present in the collated batch when outputs.loss is unavailable."
            )
        loss = loss_fn(logits, labels)

    loss.backward()

    grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    return grads, loss.item()


def _clip_gradient(grads: List[torch.Tensor], clip_norm: float):
    # Norm computed in fp32 regardless of the training dtype
    total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    clip_coef = torch.clamp(clip_norm / (total_norm + 1e-12), max=1.0)
    clipped = [(g.float() * clip_coef) for g in grads]
    return clipped, total_norm.item()


def _add_noise_and_normalize(
    running_sum: List[torch.Tensor],
    sigma: float,
    clip_norm: float,
    user_batch_size: int,
    param_dtypes: List[torch.dtype],
    generator: torch.Generator = None,
):
    """
    g_tilde_t = (1/n) * (running_sum + N(0, sigma^2 C^2 I))

    running_sum already holds sum_i g_hat_{t,i} in fp32
    Noise is generated in fp32 for numerical stability,
    then the result is cast back to each parameter's  dtype.
    """
    noisy = []
    for p_sum, dtype in zip(running_sum, param_dtypes):
        noise = torch.normal(
            mean=0.0,
            std=sigma * clip_norm,
            size=p_sum.shape,
            generator=generator,
            device=p_sum.device,
            dtype=torch.float32,
        )
        result = (p_sum + noise) / user_batch_size
        noisy.append(result.to(dtype=dtype))
    return noisy


class DPTrainer:
    """
    User-level DP trainer
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: UserLevelDataset,
        config: DPConfig,
        collator: Optional[Callable] = None,
    ):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.accountant = PrivacyAccountant(
            num_users=dataset.num_users,
            user_batch_size=config.user_batch_size,
            delta=config.delta,
        )

        self.collator = (
            collator or getattr(config, "collator", None) or _default_hf_user_collator
        )

        self.autocast_dtype = _resolve_autocast_dtype(config)

        self.rng = np.random.default_rng(self.config.seed)
        device_type = "cuda" if "cuda" in str(config.device) else "cpu"
        self.torch_generator = torch.Generator(device=device_type).manual_seed(
            self.config.seed
        )

        self._sigma = None

        # Reduces activation memory for LLM
        if getattr(self.config, "gradient_checkpointing", False):
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def _make_optimizer(
        self, trainable_params: List[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        fused_requested = getattr(self.config, "fused_optimizer", True)
        use_fused = (
            fused_requested
            and torch.cuda.is_available()
            and str(self.config.device).startswith("cuda")
        )
        try:
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                fused=use_fused,
            )
        except Exception:
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

    def train(self):

        self.model.train()

        self._sigma = self.accountant.calibrate_sigma(
            target_epsilon=self.config.target_epsilon, steps=self.config.max_steps
        )

        loss_curve: List[float] = []
        grad_norm_curve: List[float] = []
        epsilon_curve: List[float] = []
        clipped_count = 0
        processed_user = 0

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            raise ValueError("No trainable parameters found")
        param_dtypes = [p.dtype for p in trainable_params]

        optimizer = self._make_optimizer(trainable_params)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.max_steps * self.config.warmup_ratio),
            num_training_steps=self.config.max_steps,
        )

        for t in range(1, self.config.max_steps + 1):
            user_batch = self.dataset.sample_users(
                self.config.user_batch_size, self.rng
            )
            while len(user_batch) == 0:
                user_batch = self.dataset.sample_users(
                    self.config.user_batch_size, self.rng
                )

            processed_user += len(user_batch)

            # Instead of keeping every user's clipped gradient around
            running_sum = [
                torch.zeros_like(p, dtype=torch.float32) for p in trainable_params
            ]

            step_loss = 0
            step_norms = []
            for uid in user_batch:
                k_i = self.config.records_per_user.get(uid, self.config.default_k)
                records = self.dataset.sample_user_records(uid, k_i, self.rng)

                grads, loss_val = _compute_user_gradient_batched(
                    self.model,
                    self.collator,
                    records,
                    self.config.device,
                    self.autocast_dtype,
                    loss_fn=self.config.loss_fn,
                )
                clipped, pre_clip_norm = _clip_gradient(grads, self.config.clip_norm)
                clipped_count += int(pre_clip_norm > self.config.clip_norm)

                for i, g in enumerate(clipped):
                    running_sum[i] += g

                del grads, clipped

                step_loss += loss_val
                step_norms.append(pre_clip_norm)

            g_tilde = _add_noise_and_normalize(
                running_sum,
                sigma=self._sigma,
                clip_norm=self.config.clip_norm,
                user_batch_size=self.config.user_batch_size,
                param_dtypes=param_dtypes,
                generator=self.torch_generator,
            )
            del running_sum

            optimizer.zero_grad(set_to_none=True)

            for p, g in zip(trainable_params, g_tilde):
                p.grad = g

            optimizer.step()
            scheduler.step()
            eps_t = self.accountant.step()

            epsilon_curve.append(eps_t)
            loss_curve.append(step_loss / len(user_batch))
            grad_norm_curve.append(float(np.mean(step_norms)))

            if t % self.config.log_every == 0 or t == self.config.max_steps:
                print(
                    f"[DP] step {t}/{self.config.max_steps} "
                    f"loss={loss_curve[-1]:.4f} "
                    f"sigma={self._sigma:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"eps={eps_t:.4f} "
                    f"(target={self.config.target_epsilon}) "
                    f"clipped_rate={clipped_count/max(processed_user,1):.4f}"
                )

        report = self.accountant.final_report(target_epsilon=self.config.target_epsilon)
        return DPResult(
            privacy_report=report,
            loss_curve=loss_curve,
            grad_norm_curve=grad_norm_curve,
            epsilon_curve=epsilon_curve,
            sigma=self._sigma,
            clipped_rate=clipped_count / max(processed_user, 1),
        )
