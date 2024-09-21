import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "movie_title": "Avatar",  # Corrected here
    "director_name": "James Cameron",
    "actor_1_name": "CCH Pounder"
}

response = requests.post(url, json=data)

print("Response status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except ValueError:  # Handle cases where response is not JSON
    print("Response text:", response.text)
