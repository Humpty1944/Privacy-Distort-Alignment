
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class ResultManager:
    """
    Handles saving experiment results, metadata and summaries.
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, rows, filename="results.csv"):
        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path

    def save_json(self, data, filename="results.json"):
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=str)
        return path

    def save_metadata(self, config):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "configuration": config,
        }
        return self.save_json(metadata, "metadata.json")

    def save_summary(self, metrics):
        path = self.output_dir / "summary.txt"

        with open(path, "w", encoding="utf-8") as f:
            f.write("Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")

            for k, v in metrics.items():
                f.write(f"{k:<30}: {v}\n")

        return path
