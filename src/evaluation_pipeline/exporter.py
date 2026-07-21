from pathlib import Path
import json
import pandas as pd


class ResultExporter:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, result, filename="results.json"):

        with open(self.output_dir / filename, "w", encoding="utf-8") as f:

            json.dump(result, f, indent=4)

    def export_csv(self, rows, filename="results.csv"):

        pd.DataFrame(rows).to_csv(
            self.output_dir / filename,
            index=False,
        )