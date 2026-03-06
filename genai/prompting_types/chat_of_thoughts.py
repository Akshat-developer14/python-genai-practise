# Chai of thoughts

import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import os
import time

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Zero shot prompting: Directly give instructions to model along with few examples.
# Structured output
SYSTEM_PROMPT = """
You're an expert AI Assistant in resolving user queries using chain of thought
You work on START, PLAN and OUPUT steps.
You need to first PLAN what needs to be done. The PLAN can be multiple steps.
Once you think enough PLAN has been done, finally you can give an OUTPUT.

Rules:
- Strictly Follow the given JSON output format
- Only run one step at a time.
- The sequence of steps is START (where user gives an input), PLAN (That can
be multiple times) and finally OUTPUT (which is going to the displayed to
the user).

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT", "content": "string" }

Example:
START: Hey, Can you solve 2 + 3 * 5 / 10
PLAN: { "step": "PLAN": "content": "Seems like user is interested in math
problem" }
PLAN: { "step": "PLAN": "content": "looking at the problem, we should solve
this using BODMAS method" }
PLAN: { "step": "PLAN": "content": "Yes, The BODMAS is correct thing to be
done here" }
PLAN: { "step": "PLAN": "content": "first we must multiply 3 * 5 which is
15" }
PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 15 / 10" }
PLAN: { "step": "PLAN": "content": "We must perform divide that is 15 / 10
= 1.5" }
PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 1.5" }
PLAN: { "step": "PLAN": "content": "Now finally lets perform the add 3.5" }
PLAN: { "step": "PLAN": "content": "Great, we have solved and finally left
with 3.5 as ans" }
OUTPUT: { "step": "OUTPUT": "content": "3.5" }
"""

print("\n\n\n")

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]
user_query = input("👉")
message_history.append({"role": "user", "content": user_query})

while True:
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",  # Adjusted to a standard model name
            response_format={"type": "json_object"},
            messages=message_history
        )

        raw_result = response.choices[0].message.content
        # Append the assistant's thought to history so it remembers its previous plans
        message_history.append({"role": "assistant", "content": raw_result})

        try:
            parsed_result = json.loads(raw_result)  # Fixed variable name typo
        except json.JSONDecodeError:
            print("Error parsing JSON:", raw_result)
            break

        if parsed_result.get("step") == "PLAN":
            print(f"🧠 {parsed_result.get('content')}")
            continue

        if parsed_result.get("step") == "OUTPUT":
            print(f"\nResult --> {parsed_result.get('content')}")
            break
    except RateLimitError as e:
        print("Rate limit hit! Wait for 60 seconds...")
        time.sleep(60)
        continue

print("\n\n\n")