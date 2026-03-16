from typing_extensions import TypedDict, NotRequired
from typing import Optional, Literal
from langgraph.graph import StateGraph, START
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
    user_query: str
    llm_output: NotRequired[Optional[str]]
    is_good: NotRequired[Optional[bool]]

llm = ChatGroq(
    temperature=0, 
    model="openai/gpt-oss-20b",
    api_key=api_key
)

def chatbot(state: State):
    print("-> Running 20b Chatbot")
    response = llm.invoke(state.get("user_query"))
    return {"llm_output": response.content}

# NEW NODE: This evaluates the output and updates the is_good state
def grade_response(state: State):
    print("-> Grading Response")
    output = state.get("llm_output", "")
    
    # Simple logic: If the answer is short, mark it as bad (False)
    if len(output) < 50:
        print("   Result: Too short! is_good = False")
        return {"is_good": False}
    else:
        print("   Result: Good length! is_good = True")
        return {"is_good": True}

def evaluate_response(state: State) -> Literal["another_model", "__end__"]:
    print("-> Routing based on is_good:", state.get("is_good"))
    if state.get("is_good") is False:
        return "another_model"
    else:
        return "__end__"

llm2 = ChatGroq(
    temperature=0.5, 
    model="openai/gpt-oss-120b",
    api_key=api_key
)

def another_model(state: State):
    print("-> Running 120b Fallback Model")
    response = llm2.invoke(state.get("user_query"))
    return {"llm_output": response.content}

graph_builder = StateGraph(State)

# Add all three nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("grade_response", grade_response)
graph_builder.add_node("another_model", another_model)

# Define the flow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "grade_response") # Chatbot goes to grader
graph_builder.add_conditional_edges("grade_response", evaluate_response) # Grader goes to router
graph_builder.add_edge("another_model", "__end__")

graph = graph_builder.compile()

# Test with a query that will yield a short answer to force the 120b model to run
updated_state = graph.invoke({"user_query": "What is my name?"})
print("\n\nFinal State: ", updated_state)