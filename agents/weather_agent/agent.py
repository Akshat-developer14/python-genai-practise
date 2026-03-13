import json
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
import os
import time
import requests
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

"""response.json()
{'latitude': 44.4, 'longitude': 8.940001, 'generationtime_ms': 0.07486343383789062, 'utc_offset_seconds': 0, 'timezone': 'GMT', 'timezone_abbreviation': 'GMT', 'elevation': 18.0, 'current_weather_units': {'time': 'iso8601', 'interval': 'seconds', 'temperature': '°C', 'windspeed': 'km/h', 'winddirection': '°', 'is_day': '', 'weathercode': 'wmo code'}, 'current_weather': {'time': '2026-03-13T08:30', 'interval': 900, 'temperature': 13.8, 'windspeed': 10.7, 'winddirection': 14, 'is_day': 1, 'weathercode': 0}}
"""
def get_weather(city: str):
    """Fetches real-time weather using Open-Meteo (No API Key Required)."""
    
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city.strip()}&count=1&format=json"
    geo_response = requests.get(geo_url).json()

    if "results" not in geo_response or len(geo_response["results"]) == 0:
        return f"Error: Could not find coordinates for the city '{city}'."
    
    lat = geo_response["results"][0]["latitude"]
    lon = geo_response["results"][0]["longitude"]
    found_name = geo_response["results"][0]["name"]
    
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_response = requests.get(weather_url).json()
    
    # Extract exactly what we need in Python first
    temp = weather_response["current_weather"]["temperature"]
    wind = weather_response["current_weather"]["windspeed"]
    
    return f"Location: {found_name}. Temperature: {temp}°C, Windspeed: {wind} km/h."

SYSTEM_PROMPT = """
You are an expert AI Assistant using a ReAct (Reasoning and Acting) framework.
Your goal is to solve the user's query by following a strict sequence of steps: PLAN, TOOL, OBSERVE, and OUTPUT.

RULES:
- Do not guess the weather. Always use the available actions.
- Only run ONE step at a time.
- If you need data, set 'step' to 'TOOL', and provide a valid 'tool' name and 'input' string.
- DO NOT use native API function calling. You must strictly reply matching the requested JSON schema.
- Base your final OUTPUT strictly on the data provided in the OBSERVE step. Extract the temperature and windspeed to answer the user.

Available Actions You Can Request:
- get_weather: Requires a city name as input. Returns the confirmed location name and current weather data.

Output JSON Format (Strictly follow this):
- For Planning: { "step": "PLAN", "content": "detailed thought here" }
- For Action Request: { "step": "TOOL", "tool": "get_weather", "input": "city_name" }
- For Final Answer: { "step": "OUTPUT", "content": "The weather in [City] is [Temperature]°C with a wind speed of [Windspeed] km/h." }
- For Missing Data: { "step": "OUTPUT", "content": "I'm sorry, I don't have that information." }
"""

print("\n\n")

class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, TOOL, OBSERVE, OUTPUT")
    content: Optional[str] = Field(None, description="The optional string content for the step")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")

available_tools = {
    "get_weather": get_weather
}
message_history: List[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT},
]
user_query = input("👉")
message_history.append({"role": "user", "content": user_query})

while True:
    time.sleep(2)
    try:
        response = client.chat.completions.parse(
            # using openai/gpt-oss-120b, a less powerful model can't understand json and starts hallucination
            model="openai/gpt-oss-120b",
            response_format=MyOutputFormat,
            messages=message_history,
            temperature=0.1
        )

        raw_result = response.choices[0].message.content
        # Append the assistant's thought to history so it remembers its previous plans
        message_history.append({"role": "assistant", "content": raw_result})

        parsed_result = response.choices[0].message.parsed
        
        if parsed_result is None:
            print(f"❌ Error: Model failed to return the structured JSON. Raw output: {raw_result}")
            break

        if parsed_result.step == "PLAN":
            print(f"🧠: {parsed_result.content}")
            continue

        if parsed_result.step == "TOOL":
            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input

            # Checking if llm response not malformed
            if tool_to_call and tool_to_call in available_tools:

                if tool_input is None:
                    print(f"❌ Error: Tool '{tool_to_call}' requires an input, but none was provided.")
                    message_history.append({"role": "user", "content": error_msg})
                    continue

                print(f"🛠️: {tool_to_call}({tool_input})")
                tool_response = available_tools[tool_to_call](tool_input)

                message_history.append({"role": "user", "content": json.dumps({
                    'step': 'OBSERVE',
                    'tool': tool_to_call,
                    'input': tool_input,
                    'output': tool_response
                })})
            else:
                # Tool not found or malformed response (model is having hallucination)
                error_msg = f"Error: Tool '{tool_to_call}' not found or not provided."
                print(f"❌ {error_msg}")
                message_history.append({"role": "user", "content": error_msg})
            continue

        if parsed_result.step == "OUTPUT":
            print(f"\nResult --> {parsed_result.content}")
            break
    except RateLimitError as e:
        print("Rate limit hit! Wait for 60 seconds...")
        time.sleep(60)
        continue

print("\n\n\n")