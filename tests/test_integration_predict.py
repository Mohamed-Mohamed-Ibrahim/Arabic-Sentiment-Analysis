import requests


def test_prediction_positive(base_url):
    url = f"{base_url}/predict"
    payload = {
        "text":  "بصراحة التجربة كانت ممتازة وفاقت توقعاتي بكثير، جودة الخامات واضحة جداً من أول استخدام والتغليف كان محكم ونظيف وسرعة التوصيل وخدمة العملاء كانت احترافية وراقية لأبعد حد والمنتج يستاهل كل قرش بالتأكيد هكرر التعامل معاكم مرة تانية.",
    }
    
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["label"] == "Positive"

def test_prediction_negative(base_url):
    url = f"{base_url}/predict"
    payload = {
        "text": "تجربة سيئة جداً ومخيبة للآمال، المنتج وصل متأخر ومختلف تماماً عن الوصف والصور المعروضة والخامات رديئة جداً والتغليف كان ممزق وخدمة العملاء لا تستجيب تماماً لطلب الاسترجاع، ضياع وقت وفلوس على الفاضي ومستحيل أكررها.",
    }
    
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["label"] == "Negative"