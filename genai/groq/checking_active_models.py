import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

# Use json.dumps with an indent to format the output nicely
print(json.dumps(response.json(), indent=4))