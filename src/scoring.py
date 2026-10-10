import asyncio
import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_prediction_tasks: set[asyncio.Task[None]] = set()


async def save_prediction(
    text: str, label: str, confidence: float, model_version: str
) -> None:
    await asyncio.to_thread(_append_prediction, text, label, confidence, model_version)


def schedule_prediction(
    text: str, label: str, confidence: float, model_version: str
) -> None:
    task = asyncio.create_task(save_prediction(text, label, confidence, model_version))
    _prediction_tasks.add(task)

    def on_complete(completed_task: asyncio.Task[None]) -> None:
        _prediction_tasks.discard(completed_task)
        try:
            completed_task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Failed to save prediction")

    task.add_done_callback(on_complete)


def _append_prediction(
    text: str, label: str, confidence: float, model_version: str
) -> None:
    output_dir = Path("data/scoring/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "label": label,
        "confidence": float(confidence),
        "model_version": model_version,
    }

    with output_path.open("a", encoding="utf-8") as output_file:
        fcntl.flock(output_file.fileno(), fcntl.LOCK_EX)
        try:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
        finally:
            fcntl.flock(output_file.fileno(), fcntl.LOCK_UN)
