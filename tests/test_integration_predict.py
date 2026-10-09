import requests


def test_prediction_positive(base_url):
    url = f"{base_url}/predict"
    payload = {
        "text": "انا احب هذا المنتج",
        # "text": "good",
    }
    
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["label"] == "Positive"

def test_prediction_negative(base_url):
    url = f"{base_url}/predict"
    payload = {
        # "text": "انا ما راح اشتري هذا المنتج",
        "text": "string",
    }
    
    response = requests.post(url, json=payload)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["label"] == "Negative"