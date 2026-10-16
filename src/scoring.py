import asyncio
import fcntl
import json
import logging
import threading
from datetime import datetime, timezone

from config import settings

logger = logging.getLogger(__name__)
_prediction_tasks: set[asyncio.Task[None]] = set()
_prediction_count = 0
_prediction_count_lock = threading.Lock()
DRIFT_REPORT_INTERVAL = 1000


async def save_prediction(
    text: str, label: str, confidence: float, model_version: str
) -> None:
    should_generate_report = await asyncio.to_thread(
        _append_prediction, text, label, confidence, model_version
    )
    if should_generate_report:
        await asyncio.to_thread(_generate_drift_report)


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
) -> bool:
    output_dir = settings.PREDICTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    output_path = output_dir / f"predictions_{timestamp:%Y-%m-%d}.jsonl"
    record = {
        "timestamp": timestamp.isoformat(),
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

    global _prediction_count
    with _prediction_count_lock:
        _prediction_count += 1
        return _prediction_count % DRIFT_REPORT_INTERVAL == 0


def _generate_drift_report() -> None:
    from src.data_drift_evidently import main

    try:
        main()
    except Exception:
        logger.exception("Failed to generate or publish the data drift report")
