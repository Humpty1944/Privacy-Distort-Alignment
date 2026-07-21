
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """
    Generates a Markdown report summarizing an evaluation run.
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: dict,
        config: dict = None,
        filename: str = "evaluation_report.md",
    ):
        report_path = self.output_dir / filename

        lines = []
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        if config:
            lines.append("## Configuration")
            lines.append("")
            for key, value in config.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.append("## Metrics")
        lines.append("")

        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            else:
                lines.append(f"- **{key}**: {value}")

        report_path.write_text("\n".join(lines), encoding="utf-8")

        return report_path
