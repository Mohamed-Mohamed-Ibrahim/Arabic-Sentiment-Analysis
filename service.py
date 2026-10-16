from __future__ import annotations

import json
import pickle
import time
from collections import deque
from typing import Any

import bentoml
import numpy as np
import onnxruntime as ort
import scipy
import yaml
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from config import settings
from src.scoring import schedule_prediction

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

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
        from transformers import AutoTokenizer

        self.session = ort.InferenceSession(model_file_path)
        with open(vocab_file_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.classes = {0: "Negative", 1: "Positive"}
        self.version = params["version"]
        self.max_token_val = 7968

    def text_pipeline(self, text: str) -> list[int]:
        return [
            # self.vocab.get(token, 0)
            0
            if self.vocab.get(token, 0) > self.max_token_val
            else self.vocab.get(token, 0)
            for token in self.tokenizer.tokenize(str(text).lower())
        ]

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
        route="/predict",
        batchable=True,
    )
    async def predict(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        t0 = time.perf_counter()
        text_list = []
        lengths = []

        for req in requests:
            tokens = self.text_pipeline(req.text)
            if not tokens:
                tokens = [0]
            text_list.extend(tokens)
            lengths.append(len(tokens))
            production_text_lengths.append(len(req.text))

        onnx_text = np.array(text_list, dtype=np.int64)
        offsets = [0] + lengths[:-1]
        onnx_offsets = np.array(offsets, dtype=np.int64).cumsum()

        inputs = {"text": onnx_text, "offsets": onnx_offsets}
        logits = self.session.run(None, inputs)[0]

        responses = []
        preds = np.argmax(logits, axis=1)

        for idx, pred_idx in enumerate(preds):
            label = self.classes[pred_idx]
            prob = scipy.special.softmax(logits, axis=1)[idx][pred_idx]
            production_confidence_scores.append(prob)

            responses.append(
                PredictResponse(
                    label=label, confidence=prob, model_version=self.version
                )
            )
            schedule_prediction(requests[idx].text, label, prob, self.version)

        REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(
            time.perf_counter() - t0
        )
        return responses
