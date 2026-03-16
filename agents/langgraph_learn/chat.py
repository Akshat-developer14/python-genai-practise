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

print("\n")

llm = ChatGroq(
    temperature=0, 
    model="openai/gpt-oss-20b",
    api_key=api_key
)

def chatbot(state: State):# Input parameter is a state
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]} #  Returning is also a state

def samplenode(state: State):
    print("\n\nInside samplenode", state)
    return {"messages": ["Sample node."]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)

# (START) -> chatbot -> samplenode -> (END)
graph = graph_builder.compile()

updated_state = graph.invoke(State({"messages": ["Hi, my name is Akshat"]}))
print("\n\nUpdated State: ", updated_state)