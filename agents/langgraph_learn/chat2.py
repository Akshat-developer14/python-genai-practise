from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pydantic import SecretStr


load_dotenv()

raw_api_key = os.getenv("GROQ_API_KEY")
if not raw_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
api_key = SecretStr(raw_api_key)

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    temperature=0, 
    model="openai/gpt-oss-20b",
    api_key=api_key
)