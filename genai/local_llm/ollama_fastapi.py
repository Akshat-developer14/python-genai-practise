from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

# Define the structure of your request body
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat_with_llm(request: ChatRequest):
    # Use request.prompt to get the string from the JSON body
    response = ollama.chat(model='deepseek-coder:1.3b', messages=[
        {'role': 'user', 'content': request.prompt},
    ])
    return {"response": response['message']['content']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)