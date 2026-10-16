import requests


def test_health(base_url):
    url = f"{base_url}/health"
    
    response = requests.get(url)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["status"] == "healthy"