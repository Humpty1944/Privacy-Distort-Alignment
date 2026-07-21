
"""
Example demonstrating the Evaluation Pipeline.

Run:
    python examples/evaluation_pipeline_demo.py
"""

from src.evaluation_pipeline import EvaluationPipeline


def main():

    pipeline = EvaluationPipeline(output_dir="outputs")

    print("=" * 60)
    print("Loading benchmark datasets...")
    print("=" * 60)

    benchmarks = pipeline.load_benchmarks(
        n_truthfulqa=5,
        n_ifeval=5,
        n_alpaca=5,
        seed=42,
    )

    print(f"TruthfulQA : {len(benchmarks.truthfulqa)} samples")
    print(f"IFEval     : {len(benchmarks.ifeval)} samples")
    print(f"Alpaca     : {len(benchmarks.alpaca)} samples")

    metrics = {
        "truthfulness": 0.82,
        "hallucination_rate": 0.14,
        "instruction_following": 0.88,
        "helpfulness": 0.79,
        "refusal_accuracy": 1.00,
        "false_refusal_rate": 0.05,
    }

    rows = [metrics]

    config = {
        "model": "EleutherAI/pythia-70m",
        "seed": 42,
        "truthfulqa": 5,
        "ifeval": 5,
        "alpaca": 5,
    }

    pipeline.save_results(
        rows=rows,
        metrics=metrics,
        config=config,
    )

    print("\nOutputs generated in ./outputs/")
    print(" - results.csv")
    print(" - results.json")
    print(" - metadata.json")
    print(" - summary.txt")
    print(" - evaluation_report.md")

    print("\nEvaluation Pipeline demo completed successfully.")


if __name__ == "__main__":
    main()
