from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model import ArabicSentimentModel
from fastapi.responses import RedirectResponse


app = FastAPI(title="Ride Duration API")
model = ArabicSentimentModel()


# ── Schemas ─────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=4, max_length=2000)


class PredictResponse(BaseModel):
    result: str


# ── Handlers ────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    res = model.predict(req.text)
    return PredictResponse(result=res)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
