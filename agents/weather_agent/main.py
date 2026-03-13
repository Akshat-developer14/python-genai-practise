from dotenv import load_dotenv
from openai import OpenAI
import os
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

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
    
    return f"Location: {found_name}. Weather data: {weather_response}"
        
        

def main():
    user_query = input("> ")
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite-preview",
        messages=[
            {'role': 'user', 'content': user_query}
        ]
    )

    print(f'🤖: {response.choices[0].message.content}')

main()