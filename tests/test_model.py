# from src.model import ArabicSentimentModel

# def test_create_user(base_url):
#     # Arrange: Set up headers and the payload
#     url = f"{base_url}/users"
#     payload = {
#         "name": "morpheus",
#         "job": "leader"
#     }
    
#     # Act: Send the POST request
#     response = requests.post(url, json=payload)
    
#     # Assert: Validate the creation details
#     assert response.status_code == 201
    
#     response_data = response.json()
#     assert response_data["name"] == payload["name"]
#     assert "id" in response_data  # Ensure an ID was created