from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()

# api key
raw_api_key = os.getenv("GROQ_API_KEY")
if not raw_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
api_key = SecretStr(raw_api_key)

# vector embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
)

# chat model
llm = ChatGroq(
    temperature=0, 
    model="openai/gpt-oss-120b",
    api_key=api_key
)

# Database
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)
# Take user input 
user_query = input("\nAsk something from PDF: ")

# Similarity search (Relevant chunks from vector db)
search_results = vector_db.similarity_search(query=user_query, k=3)

context = "\n\n\n".join([
    f"--- Chunk Start ---\n"
    f"Page: {res.metadata.get('page_label', 'N/A')}\n"
    f"Source: {res.metadata.get('source', 'Unknown')}\n"
    f"Content: {res.page_content}\n"
    f"--- Chunk End ---" 
    for res in search_results
])

SYSTEM_PROMPT = f'''
You are a helpful Ai Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

Context:
{context}
'''

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=user_query),
]

# Get the Response
print("\nThinking...")
response = llm.invoke(messages)

print("-" * 30)
print(f"Assistant: {response.content}")
print("-" * 30)
