from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite-preview",
    messages=[
        {"role": "user", "content": "Hey, there i am akshat"}
    ]
)

# Remember: OpenAI SDK uses choices[0].message.content
print(response.choices[0].message.content)