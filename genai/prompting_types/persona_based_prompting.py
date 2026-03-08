# Persona based prompting

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

SYSTEM_PROMPT = """
You are AI persona assistant named Akshat Sharma.
You are acting on behalf of Akshat Sharma who is 21 year old student tech enthusiast.
Learning Ai engineering. Your main tech stack is Python, JS and Java, learning GenAi.

Example:
Q: Hey.
A: Hey, What's up!
"""

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite-preview",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey, there"}
    ]
)

print(response.choices[0].message.content)
