from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (build_safety_probes, load_corpus, probe_training_text,
                      load_truthfulqa_probes, load_ifeval_probes, load_alpaca_probes)
from src.embeddings import WordEmbeddings
from src.privacy_mechanisms import get_mechanism
from src.semantic_distortion import input_text_distortion, representation_distortion
from src.sentence_encoder import get_sentence_encoder
from src.alignment_metrics import evaluate_all
from src.decomposition import decompose_pathways, mediation_analysis, plot_privacy_alignment_curves

OUT_DIR = Path(__file__).parent / "outputs"

TEXT_LEVEL_MECHANISMS = ["no_privacy", "random_token_perturbation", "semantic_sanitization"]
ALL_MECHANISMS = TEXT_LEVEL_MECHANISMS + ["dp_sgd_finetune"]

ALIGNMENT_METRIC_NAMES = [
    "truthfulness", "hallucination_rate", "refusal_accuracy",
    "false_refusal_rate", "helpfulness", "instruction_following",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-model-name", default=None, help="Defaults to hf_model.DEFAULT_MODEL_NAME")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1.0, 3.0, 8.0])
    ap.add_argument("--dp-delta", type=float, default=1e-5)
    ap.add_argument("--dp-clip-norm", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                     help="Gradient clipping for plain (non-DP) fine-tuning.")
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                     help="Fraction of plain-training steps to linearly warm up the LR over "
                          "(0 -> lr).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs (plain training)")
    ap.add_argument("--dp-epochs", type=int, default=20,
                     help="DP-SGD epochs -- higher than --epochs by default since a much larger "
                          "--dp-batch-size means far fewer gradient steps per epoch")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dp-batch-size", type=int, default=16,
                     help="DP-SGD batch size.")
    ap.add_argument("--dp-max-physical-batch-size", type=int, default=8,
                     help="What actually gets run through the GPU at once for DP-SGD, via "
                          "Opacus's BatchMemoryManager -- decoupled from --dp-batch-size (the "
                          "logical/privacy-relevant batch size) since Opacus needs a per-example "
                          "gradient tensor per parameter, a much larger memory footprint than "
                          "normal training.")
    ap.add_argument("--emb-dim", type=int, default=16, help="Dim for the semantic-sanitization embeddings")
    ap.add_argument("--semantic-noise-scale", type=float, default=0.05)
    ap.add_argument("--max-probes", type=int, default=None, help="Truncate probe set.")
    ap.add_argument("--n-truthfulqa", type=int, default=20,
                     help="Real TruthfulQA questions for truthfulness/hallucination")
    ap.add_argument("--n-ifeval", type=int, default=15,
                     help="Real IFEval prompts for instruction_following")
    ap.add_argument("--n-alpaca", type=int, default=15,
                     help="Real Alpaca instruction/output pairs for helpfulness")
    ap.add_argument("--alpaca-max-target-chars", type=int, default=0,
                     help="Optionally exclude Alpaca answers longer than this (chars) from "
                          "sampling.")
    ap.add_argument("--eval-holdout-frac", type=float, default=0.0,
                     help="Fraction of TruthfulQA questions / Alpaca examples to hold out from "
                          "fine-tuning entirely (eval-only), to measure generalization to unseen "
                          "examples rather than just faithful-learning of the fine-tuning set. "
                          "Default 0.0 = fine-tune and evaluate on the same examples.")
    ap.add_argument("--no-sentence-encoder", action="store_true",
                     help="Skip loading a real sentence-transformers model for "
                          "embedding_similarity; fall back to mean-pooled word embeddings")
    return ap.parse_args()


def main():
    args = parse_args()
    from src.hf_model import HFModel, DEFAULT_MODEL_NAME
    from src.hf_training import train_dp, train_plain

    model_name = args.hf_model_name or DEFAULT_MODEL_NAME
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"[1/5] Loading base model {model_name} ...")
    cond_model = HFModel(model_name=model_name, seed=0)
    base_state = {k: v.clone() for k, v in cond_model.model.state_dict().items()}
    ref_model = HFModel(model_name=model_name, seed=0)

    print("[2/5] Building probe set (real TruthfulQA + IFEval + Alpaca, toy safety) "
          "+ embeddings for semantic sanitization ...")
    seed0 = args.seeds[0] if args.seeds else 0
    safety_probes = build_safety_probes()
    real_truthfulqa = load_truthfulqa_probes(n=args.n_truthfulqa, seed=seed0,
                                              eval_holdout_frac=args.eval_holdout_frac)
    real_ifeval = load_ifeval_probes(n=args.n_ifeval, seed=seed0)
    real_alpaca = load_alpaca_probes(
        n=args.n_alpaca, seed=seed0, eval_holdout_frac=args.eval_holdout_frac,
        max_target_chars=(args.alpaca_max_target_chars or None))

    probes = real_truthfulqa + real_ifeval + real_alpaca + safety_probes
    print(f"    Using: {len(safety_probes)} toy safety, {len(real_truthfulqa)} real TruthfulQA, "
          f"{len(real_ifeval)} real IFEval, {len(real_alpaca)} real Alpaca probes ({len(probes)} total).")
    if args.max_probes:
        probes = probes[: args.max_probes]

    finetune_probes = [p for p in probes if p.target and not p.held_out]
    probe_texts = probe_training_text(finetune_probes)
    eval_prompts = [p.prompt for p in probes]
    n_held_out = sum(p.held_out for p in probes)
    print(f"    {len(finetune_probes)}/{len(probes)} probes have a target and are used for "
          f"fine-tuning ({n_held_out} held out from fine-tuning for eval-only); "
          f"all {len(probes)} are used for evaluation.")
    corpus = load_corpus()
    embeddings = WordEmbeddings(dim=args.emb_dim, min_count=2).fit(
        corpus + " " + (" ".join(probe_texts) + " ") * 30)
    vocab_words = embeddings.vocab
    sentence_encoder = None if args.no_sentence_encoder else get_sentence_encoder()

    rows = []
    n_conditions = len(ALL_MECHANISMS) * len(args.epsilons) * len(args.seeds)
    print(f"[3/5] Running {n_conditions} (mechanism x epsilon x seed) fine-tuning conditions ...")
    run_i = 0
    for seed in args.seeds:
        rng = np.random.default_rng(seed)

        cond_model.model.load_state_dict(base_state)
        ref_log = train_plain(cond_model, probe_texts, epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size, max_grad_norm=args.max_grad_norm,
                               warmup_frac=args.warmup_frac, seed=seed)
        ref_model.model.load_state_dict(cond_model.model.state_dict())

        for mech_name in ALL_MECHANISMS:
            for epsilon in args.epsilons:
                run_i += 1

                if mech_name == "no_privacy":
                    cond_model.model.load_state_dict(ref_model.model.state_dict())
                    text_distortion = 0.0
                    log = ref_log
                    eval_model = cond_model
                elif mech_name in TEXT_LEVEL_MECHANISMS:
                    cond_model.model.load_state_dict(base_state)
                    mech = get_mechanism(mech_name, vocab_words, embeddings,
                                          semantic_noise_scale=args.semantic_noise_scale)
                    privatized_texts = [mech.privatize(t, epsilon, rng) for t in probe_texts]
                    log = train_plain(cond_model, privatized_texts, epochs=args.epochs, lr=args.lr,
                                       batch_size=args.batch_size, max_grad_norm=args.max_grad_norm,
                                       warmup_frac=args.warmup_frac, seed=seed)
                    dists = [input_text_distortion(orig, priv, embeddings, sentence_encoder)["input_text_distortion"]
                             for orig, priv in zip(probe_texts, privatized_texts)]
                    text_distortion = float(np.mean(dists))
                    eval_model = cond_model
                else:
                    dp_model = HFModel(model_name=model_name, seed=seed)
                    dp_model.model.load_state_dict(base_state)
                    log = train_dp(dp_model, probe_texts, epsilon=epsilon, delta=args.dp_delta,
                                    clip_norm=args.dp_clip_norm, epochs=args.dp_epochs, lr=args.lr,
                                    batch_size=args.dp_batch_size,
                                    max_physical_batch_size=args.dp_max_physical_batch_size, seed=seed)
                    text_distortion = 0.0
                    eval_model = dp_model

                metrics = evaluate_all(eval_model.generate_fn(), probes)
                ece = eval_model.calibration_ece(probes)
                rep_distortion = representation_distortion(ref_model.predict_fn(), eval_model.predict_fn(),
                                                             eval_prompts)

                rows.append({
                    "mechanism": mech_name,
                    "epsilon": epsilon,
                    "seed": seed,
                    "input_text_distortion": text_distortion,
                    "representation_distortion": rep_distortion,
                    "calibration_ece": ece,
                    "loss_variance": log.loss_variance if log is not None else float("nan"),
                    "mean_grad_norm": log.mean_grad_norm if log is not None else float("nan"),
                    "mean_clip_rate": log.mean_clip_rate if log is not None else float("nan"),
                    "noise_multiplier": log.noise_multiplier if log is not None else float("nan"),
                    **metrics,
                })
                print(f"    ... {run_i}/{n_conditions} done ({time.time() - t0:.1f}s elapsed): "
                      f"{mech_name} eps={epsilon} -> truthfulness={metrics['truthfulness']:.2f}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[4/5] Saved raw results -> {csv_path}")

    print("[5/5] Running decomposition analysis + generating curves ...")
    text_mech_df = df[df["mechanism"].isin(TEXT_LEVEL_MECHANISMS)]
    report = {}
    for metric in ALIGNMENT_METRIC_NAMES:
        report[metric] = {
            "mediation_via_representation_distortion": mediation_analysis(
                df, outcome=metric, treatment="epsilon", mediator="representation_distortion"),
            "pathway_decomposition": decompose_pathways(
                df, outcome=metric,
                predictors=["epsilon", "representation_distortion", "calibration_ece", "mean_grad_norm"]),
            "text_mechanism_mediation_via_input_text_distortion": mediation_analysis(
                text_mech_df, outcome=metric, treatment="epsilon", mediator="input_text_distortion"),
            "text_mechanism_pathway_decomposition": decompose_pathways(
                text_mech_df, outcome=metric,
                predictors=["epsilon", "input_text_distortion", "representation_distortion", "calibration_ece"]),
            "dp_sgd_pathway_decomposition": decompose_pathways(
                df[df["mechanism"] == "dp_sgd_finetune"], outcome=metric,
                predictors=["epsilon", "representation_distortion", "calibration_ece",
                            "mean_grad_norm", "mean_clip_rate", "noise_multiplier"]),
        }
        try:
            plot_privacy_alignment_curves(df, metric, str(OUT_DIR / f"curve_{metric}.png"))
        except Exception as e:
            print(f"    [warn] could not plot {metric}: {e}")

    (OUT_DIR / "decomposition_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
