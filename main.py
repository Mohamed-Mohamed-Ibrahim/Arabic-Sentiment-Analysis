from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.model import ArabicSentimentModel

app = FastAPI(title="Ride Duration API")
model = ArabicSentimentModel()


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
    res, prob, version = await run_in_threadpool(model.predict, req.text)
    return PredictResponse(label=res, confidence=round(prob, 5), model_version=version)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
