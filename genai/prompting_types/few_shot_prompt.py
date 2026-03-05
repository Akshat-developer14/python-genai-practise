# Few shot prompting

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Zero shot prompting: Directly give instructions to model along with few examples.
# Structured output
SYSTEM_PROMPT = """You should only and only ans the coding related questions. Do not ans anything else. Your name is Alexa. If user asks something other than coding, just say sorry.

Rule:
- Strictly follow the output in JSON format

Output Format:
{{
"code": "string",
"isCodingQuestion": boolean
}}

Example:
Q: Can you explain the a + b whole Square?
A: {{ "code": "string", "isCodingQuestion": false }}

Q: Hey, write a code in python add two numbers.
A: {{
"code": "def add(a, b):
    return a + b",
"isCodingQuestion": boolean
}}

"""

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite-preview",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Can you write a swap two numbers without using third variable code in java also keep main and function class separate?"}
    ]
)

# Remember: OpenAI SDK uses choices[0].message.content
print(response.choices[0].message.content)