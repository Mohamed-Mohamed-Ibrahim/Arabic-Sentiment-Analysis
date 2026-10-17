from __future__ import annotations

import json
import time
from collections import deque
from typing import Any

import bentoml
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from config import settings
from src.model import ArabicSentimentModel as SentimentModel
from src.scoring import schedule_prediction

DRIFT_GAUGE = Gauge("feature_psi_score", "PSI drift score per feature", ["feature"])
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "End-to-end API latency", ["endpoint", "status"]
)

production_text_lengths = deque(maxlen=1000)
production_confidence_scores = deque(maxlen=1000)

with settings.BASELINE_PATH.open(encoding="utf-8") as baseline_file:
    training_baseline = json.load(baseline_file)
TRAINING_TEXT_LENGTHS = training_baseline["text_lengths"]
TRAINING_CONFIDENCE_SCORES = training_baseline["confidence_scores"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=4, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@bentoml.service(
    name=settings.BENTOML_SERVICE_NAME,
    resources={"cpu": "2"},
    http={"port": settings.API_PORT},
)
class ArabicSentimentModel:
    def __init__(
        self,
        tokenizer_model_name=settings.MODEL_TOKENIZER,
        vocab_file_path=str(settings.MODEL_VOCAB_PATH),
        model_file_path=str(settings.MODEL_ONNX_PATH),
    ):
        self.model = SentimentModel(
            tokenizer_model_name=tokenizer_model_name,
            vocab_file_path=vocab_file_path,
            model_file_path=model_file_path,
        )

    @bentoml.api(route="/health")
    def health(self) -> dict[str, str]:
        return {"status": "healthy"}

    @bentoml.api(route="/internal/gauges")
    def publish_gauges(self, payload: dict[str, Any]) -> dict[str, int]:
        n = 0
        for feature, value in (payload.get("feature_psi") or {}).items():
            DRIFT_GAUGE.labels(feature=feature).set(float(value))
            n += 1
        return {"published": n}

    @bentoml.api(route="/metrics")
    def metrics(self) -> Any:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @bentoml.api(
        route="/predict/batch",
        batchable=True,
    )
    async def predict_batch(
        self, requests: list[PredictRequest]
    ) -> list[PredictResponse]:
        t0 = time.perf_counter()
        responses = []
        for req in requests:
            label, confidence, version = await run_in_threadpool(
                self.model.predict, [req.text]
            )
            production_text_lengths.append(len(req.text))
            production_confidence_scores.append(confidence)
            responses.append(
                PredictResponse(
                    label=label,
                    confidence=round(confidence, 5),
                    model_version=version,
                )
            )
            schedule_prediction(req.text, label, confidence, version)

        REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(
            time.perf_counter() - t0
        )
        return responses

    @bentoml.api(route="/predict")
    async def predict_one(self, req: PredictRequest) -> PredictResponse:
        t0 = time.perf_counter()
        label, confidence, version = await run_in_threadpool(
            self.model.predict, [req.text]
        )
        schedule_prediction(req.text, label, confidence, version)
        production_text_lengths.append(len(req.text))
        production_confidence_scores.append(confidence)
        REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(
            time.perf_counter() - t0
        )
        return PredictResponse(
            label=label, confidence=round(confidence, 5), model_version=version
        )
