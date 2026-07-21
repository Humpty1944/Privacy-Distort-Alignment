
from .benchmark_loader import BenchmarkLoader
from .result_manager import ResultManager
from .report_generator import ReportGenerator


class EvaluationPipeline:
    """
    High-level interface for loading benchmarks,
    storing experiment outputs and generating reports.
    """

    def __init__(self, output_dir="outputs"):

        self.loader = BenchmarkLoader()
        self.results = ResultManager(output_dir)
        self.report = ReportGenerator(output_dir)

    def load_benchmarks(
        self,
        n_truthfulqa=20,
        n_ifeval=15,
        n_alpaca=15,
        seed=0,
    ):

        return self.loader.load_all(
            n_truthfulqa=n_truthfulqa,
            n_ifeval=n_ifeval,
            n_alpaca=n_alpaca,
            seed=seed,
        )

    def save_results(
        self,
        rows,
        metrics,
        config,
    ):

        self.results.save_csv(rows)
        self.results.save_json(metrics)
        self.results.save_metadata(config)
        self.results.save_summary(metrics)

        self.report.generate(
            metrics=metrics,
            config=config,
        )

        return True
