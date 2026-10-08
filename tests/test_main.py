import pytest

from main import PredictRequest, health, predict


@pytest.mark.asyncio
async def test_predict_positive(base_url):

    # req = PredictRequest(text="good")
    req = PredictRequest(text="انا احب هذا المنتج")

    data = await predict(req)

    assert data.result == "Positive"

@pytest.mark.asyncio
async def test_predict_negative(base_url):

    req = PredictRequest(text="string")
    # req = PredictRequest(text="انا ما راح اشتري هذا المنتج")

    data = await predict(req)

    assert data.result == "Negative"

@pytest.mark.asyncio
async def test_health(base_url):

    data = await health()

    assert data["status"] == "healthy"
