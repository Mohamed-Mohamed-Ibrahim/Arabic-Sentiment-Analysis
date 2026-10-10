import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

BASELINE_PATH = Path("data/training_baseline.json")
PREDICTIONS_DIR = Path("data/scoring/output")
REPORT_PATH = Path("reports/drift_report.html")
GAUGES_URL = os.environ.get("GAUGES_URL", "http://localhost:8000/internal/gauges")


def load_reference(path: Path = BASELINE_PATH) -> pd.DataFrame:
    with path.open(encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)

    text_lengths = baseline["text_lengths"]
    confidence_scores = baseline["confidence_scores"]
    if len(text_lengths) != len(confidence_scores):
        raise ValueError("Baseline feature arrays must have the same number of samples")

    return pd.DataFrame(
        {
            "text_length": text_lengths,
            "confidence_score": confidence_scores,
        }
    )


def _earliest_predictions_file(path: Path) -> Path:
    if path.is_file():
        return path

    dated_files = sorted(path.glob("predictions_????-??-??.jsonl"))
    if dated_files:
        return dated_files[0]
    
    raise FileNotFoundError(f"No prediction JSONL files found in {path}")


def load_current(path: Path = PREDICTIONS_DIR, limit: int = 1000) -> pd.DataFrame:
    predictions_path = _earliest_predictions_file(path)
    records: deque[dict[str, Any]] = deque(maxlen=limit)
    with predictions_path.open(encoding="utf-8") as predictions_file:
        for line_number, line in enumerate(predictions_file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {predictions_path}"
                    ) from error

    if not records:
        raise ValueError(f"No prediction records found in {predictions_path}")

    return pd.DataFrame(
        {
            "text_length": [len(str(record["text"])) for record in records],
            "confidence_score": [float(record["confidence"]) for record in records],
        }
    )


def build_report(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[Report, dict[str, float]]:
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="text_length", stattest="psi"),
            ColumnDriftMetric(column_name="confidence_score", stattest="psi"),
        ]
    )
    report.run(reference_data=reference, current_data=current)

    feature_psi = {
        metric["result"]["column_name"]: float(metric["result"]["drift_score"])
        for metric in report.as_dict()["metrics"]
    }
    return report, feature_psi


def main() -> None:
    reference = load_reference()
    current = load_current()
    report, feature_psi = build_report(reference, current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    response = requests.post(
        GAUGES_URL,
        json={"feature_psi": feature_psi},
        timeout=10,
    )
    response.raise_for_status()
    print(f"Saved drift report to {REPORT_PATH}")
    print(f"Published PSI scores: {feature_psi}")


if __name__ == "__main__":
    main()
