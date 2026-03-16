from typing_extensions import TypedDict
from typing import Annotated
from langgraph.checkpoint.mongodb import MongoDBSaver
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
    model="openai/gpt-oss-120b",
    api_key=api_key
)


def chatbot(state: State):# Input parameter is a state
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]} #  Returning is also a state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
    

DB_URI = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer=checkpointer)

    config = {
        "configurable":{
            "thread_id": "Akshat"
        }
    }
    for chunk in graph_with_checkpointer.stream(
        State({"messages": ["i was checking memory and saw langgraph stored our conversation in encrypted format"]}),
        config,
        stream_mode="values"
        ):
        chunk["messages"][-1].pretty_print()
