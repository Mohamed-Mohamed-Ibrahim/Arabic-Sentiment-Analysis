import pytest

from main import PredictRequest, health, predict


@pytest.mark.asyncio
async def test_predict_positive(base_url):

    req = PredictRequest(text="بصراحة التجربة كانت ممتازة وفاقت توقعاتي بكثير، جودة الخامات واضحة جداً من أول استخدام والتغليف كان محكم ونظيف وسرعة التوصيل وخدمة العملاء كانت احترافية وراقية لأبعد حد والمنتج يستاهل كل قرش بالتأكيد هكرر التعامل معاكم مرة تانية.")

    data = await predict(req)

    assert data.label == "Positive"


@pytest.mark.asyncio
async def test_predict_negative(base_url):

    req = PredictRequest(text= "تجربة سيئة جداً ومخيبة للآمال، المنتج وصل متأخر ومختلف تماماً عن الوصف والصور المعروضة والخامات رديئة جداً والتغليف كان ممزق وخدمة العملاء لا تستجيب تماماً لطلب الاسترجاع، ضياع وقت وفلوس على الفاضي ومستحيل أكررها.")

    data = await predict(req)

    assert data.label == "Negative"


@pytest.mark.asyncio
async def test_health(base_url):

    data = await health()

    assert data["status"] == "healthy"
