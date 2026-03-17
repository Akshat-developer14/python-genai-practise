from mem0 import Memory
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=groq_api_key
)

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434", # Optional, but highly recommended
            "embedding_dims": 768 # Required for nomic-embed-text to prevent database mismatch
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.3-70b-versatile",
            "api_key": groq_api_key
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "testing_mem0",
            "embedding_model_dims": 768
        }
    }
}

mem_client = Memory.from_config(config)

while True:
    user_query = input("You: ")

    search_memory = mem_client.search(
        user_id="akshat",
        query=user_query,
        limit=3
    )

    memories = [
        f"ID: {mem.get("id")}\nMemory: {mem.get('memory')}"
        for mem in search_memory.get("results")
    ]
    
    SYSTEM_PROMPT = f'''
        Here is context about user:
        {json.dumps(memories)}
    '''

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    ai_response = response.choices[0].message.content
    print(f"AI: {ai_response}")
    mem_client.add(
        user_id="akshat",
        messages=[
            {
                "role": "user",
                "content": user_query,
            },
            {
                "role": "assistant",
                "content": ai_response,
            }
        ]
    )
    print("Memory has been saved.....")