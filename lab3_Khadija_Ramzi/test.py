# test.py

import requests

url = "http://127.0.0.1:8000/predict"

sample = {
    "bill_length_mm": 42.0,
    "bill_depth_mm": 17.0,
    "flipper_length_mm": 200,
    "body_mass_g": 4100,
    "year": 2009,
    "sex": "male",
    "island": "Biscoe"
}

response = requests.post(url, json=sample)

print("Status Code:", response.status_code)
print("Prediction:", response.json())
