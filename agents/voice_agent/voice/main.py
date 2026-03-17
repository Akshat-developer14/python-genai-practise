import speech_recognition as sr
from kokoro_onnx import Kokoro
import sounddevice as sd
from dotenv import load_dotenv
from groq import Groq
import os
import time
import sys

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

VOICE_FILE = "kokoro-v1.0.onnx"
VOICES_BIN = "voices-v1.0.bin"

if not os.path.exists(VOICE_FILE) or not os.path.exists(VOICES_BIN):
    print(f"❌ ERROR: Missing model files in {os.getcwd()}")
    sys.exit(1)

tts = Kokoro(VOICE_FILE, VOICES_BIN)

recognizer = sr.Recognizer() # Speech to text

def main():
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            # 1. Sensitivity: Higher = ignores more background noise (0-4000)
            # Since you have a fan, try 1000 or 1500.
            recognizer.energy_threshold = 1000 
            
            # 2. Speed: How many seconds of silence before it stops recording.
            # Default is 0.8. Let's drop it to 0.5 for a "fast" feel.
            recognizer.pause_threshold = 0.8
            
            # 3. Dynamic adjustment: Let it adjust to your room's noise floor
            # recognizer.dynamic_energy_threshold = True

            print("Speak now...")
            audio = recognizer.listen(source)

            print("Processing Audio... (STT)")
            try:
                stt = recognizer.recognize_google(audio)  # type: ignore[attr-defined]
                print(f"You: {stt}")
                if "exit" in stt.lower():
                    print("Goodbye!")
                    break
            except sr.UnknownValueError:
                print("Could not understand audio, please try again.")
                continue
            except sr.RequestError as e:
                print(f"Google STT service error: {e}")

            SYSTEM_PROMPT = """You are a export voice assistant. You are given the transcript of what user has said using voice. You need to output as if you are an voice agent and whatever you speak will be converted back to voice using AI and played back to user.
            Most important as an voice assistant you should be very concise and to the point, but if user asks to write a code don't do that and say "Sorry, i am just a voice assistant" because text-to-speech model will not be able to pronounce the code or such a big text.
                     """
            try:
                print("Agent is thinking...")
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT
                        },
                        {"role": "user", "content": stt},
                    ],
                )
            except Exception as e:
                print(f"Groq API error: {e}")
                continue
            reply = response.choices[0].message.content
            if reply is None:
                print("Groq API failed to return a response. Please try again.")
                continue
            print(f"Agent: {reply}")
            samples, sample_rate = tts.create(reply, voice="af_bella", speed=0.95)
            sd.play(samples, sample_rate)
            sd.wait()
            time.sleep(2)

if __name__ == "__main__":
    main()