import json
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from src.model import ArabicSentimentModel
from src.scoring import schedule_prediction

app = FastAPI(title=settings.APP_NAME)
model = ArabicSentimentModel(
    tokenizer_model_name=settings.MODEL_TOKENIZER,
    vocab_file_path=str(settings.MODEL_VOCAB_PATH),
    model_file_path=str(settings.MODEL_ONNX_PATH),
)

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


# ── Schemas ─────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=4, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


# ── Handlers ────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.perf_counter()
    res, prob, version = await run_in_threadpool(model.predict, [req.text])
    schedule_prediction(req.text, res, prob, version)
    production_text_lengths.append(len(req.text))
    production_confidence_scores.append(prob)
    REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(
        time.perf_counter() - t0
    )
    return PredictResponse(label=res, confidence=round(prob, 5), model_version=version)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/internal/gauges")
def publish_gauges(payload: dict[str, Any]) -> dict[str, int]:
    n = 0
    for feature, value in (payload.get("feature_psi") or {}).items():
        DRIFT_GAUGE.labels(feature=feature).set(float(value))
        n += 1
    return {"published": n}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
