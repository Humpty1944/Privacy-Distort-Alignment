# Evaluation Report

Generated: 2026-07-21T21:50:47.358688

## Configuration

- **hf_model_name**: None
- **epsilons**: [0.5, 1.0, 3.0, 8.0]
- **dp_delta**: 1e-05
- **dp_clip_norm**: 1.0
- **max_grad_norm**: 1.0
- **warmup_frac**: 0.1
- **seeds**: [0]
- **epochs**: 5
- **dp_epochs**: 20
- **lr**: 5e-05
- **batch_size**: 8
- **dp_batch_size**: 16
- **dp_max_physical_batch_size**: 8
- **emb_dim**: 16
- **semantic_noise_scale**: 0.05
- **max_probes**: None
- **n_truthfulqa**: 2
- **n_ifeval**: 2
- **n_alpaca**: 2
- **alpaca_max_target_chars**: 0
- **eval_holdout_frac**: 0.0
- **no_sentence_encoder**: False

## Metrics

- **epsilon**: 3.1250
- **seed**: 0.0000
- **input_text_distortion**: 0.0808
- **representation_distortion**: inf
- **calibration_ece**: 0.0635
- **loss_variance**: 1.0037
- **mean_grad_norm**: 69.8619
- **mean_clip_rate**: 1.0000
- **noise_multiplier**: 14.5782
- **truthfulness**: 0.2500
- **hallucination_rate**: 0.0000
- **refusal_accuracy**: 0.5781
- **false_refusal_rate**: 0.5938
- **helpfulness**: 0.0609
- **instruction_following**: 0.4375