import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

headers = {
    "Api-Key": api_key,
    "Accept": "application/json"
}

response = requests.get(
    "https://controller.pinecone.io/indexes",
    headers=headers
)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")