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
            "model": "openai/gpt-oss-120b",
            "api_key": groq_api_key
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://f4a87830.databases.neo4j.io",
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD")
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

chat_history = [] 

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in ["/exit", "\\exit", "exit", "quit"]:
        print("Jarvis: Shutting down. Goodbye, Akshat.")
        break

    if not user_query:
        continue

    # 1. Search Long-Term Vector Memory (Qdrant)
    search_memory = mem_client.search(
        user_id="akshat",
        query=user_query,
        limit=3
    )
    
    vector_memories = [
        f"- {mem.get('memory')}" 
        for mem in search_memory.get("results", [])
    ]
    vector_context = "\n".join(vector_memories) if vector_memories else "No relevant vector memories found."

    # 2. Search Long-Term Graph Memory (Neo4j)
    graph_context = "No relevant graph relationships found."
    if mem_client.graph: 
        try:
            graph_search = mem_client.graph.search(
                query=user_query,
                user_id={"user_id": "akshat"}
            )
            
            graph_memories = []
            if graph_search:
                for item in graph_search:
                    if isinstance(item, dict) and 'source' in item and 'relationship' in item and 'target' in item:
                        graph_memories.append(f"- {item['source']} -> {item['relationship']} -> {item['target']}")
                    else:
                        graph_memories.append(f"- {str(item)}")
                
                if graph_memories:
                    graph_context = "\n".join(graph_memories)
                    
        except Exception as e:
            print(f"[Debug] Graph search error: {e}")

    # 3. Build the Ultimate System Prompt
    SYSTEM_PROMPT = f'''
    You are a personal assistant named Jarvis. 
    Here is relevant long-term context about the user retrieved from your memory databases:
    
    [Semantic Facts (Vector DB)]:
    {vector_context}

    [Entity Relationships (Graph DB)]:
    {graph_context}

    Use this combined context to understand the user's world and personalize your response. If the context isn't relevant to the current question, ignore it.
    '''

    # 4. Build the full conversation thread
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    api_messages.extend(chat_history)
    api_messages.append({"role": "user", "content": user_query})

    # 5. Call the LLM
    response = client.chat.completions.create(
        messages=api_messages,
        model="openai/gpt-oss-120b",
    )
    
    ai_response = response.choices[0].message.content
    print(f"AI: {ai_response}")

    # 6. Update Short-Term Memory
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": ai_response})
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # 7. Save to Long-Term Memory (Extracts vectors for Qdrant and nodes for Neo4j)
    mem_client.add(
        user_id="akshat",
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response}
        ]
    )