from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model import ArabicSentimentModel
from fastapi.responses import RedirectResponse


app = FastAPI(title="Ride Duration API")
model = ArabicSentimentModel()


# ── Schemas ─────────────────────────────────────────
class PredictRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    passengers: int = Field(1, ge=1, le=8)


class PredictResponse(BaseModel):
    duration_min: float


# ── Handlers ────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    d = model.predict([req.distance_km, req.passengers])
    return PredictResponse(duration_min=round(d, 2))


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
