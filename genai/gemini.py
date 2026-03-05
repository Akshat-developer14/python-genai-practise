from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview", contents="Explain why ai general intelligence will come in 5 or 10 years is a myth in short."
)
print(response.text)