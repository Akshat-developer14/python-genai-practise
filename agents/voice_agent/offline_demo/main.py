import os
import sys
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
# You can use 'llama-3.3-70b-versatile' for high intelligence 
# or 'llama3-8b-8192' for extreme speed.
GROQ_MODEL = "openai/gpt-oss-120b" 
STT_MODEL_SIZE = "base.en"
VOICE_FILE = "kokoro-v1.0.onnx"
VOICES_BIN = "voices-v1.0.bin"

# IMPORTANT: Set your API key in your terminal before running:
# Windows (CMD): set GROQ_API_KEY=your_key_here
# Windows (PowerShell): $env:GROQ_API_KEY="your_key_here"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

if not os.path.exists(VOICE_FILE) or not os.path.exists(VOICES_BIN):
    print(f"❌ ERROR: Missing model files in {os.getcwd()}")
    sys.exit(1)

# --- 2. INITIALIZE MODELS ---
print("Loading Local Models (STT & TTS)...")
stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
tts = Kokoro(VOICE_FILE, VOICES_BIN)

def record_audio(duration=5, fs=16000):
    print("\n[Listening...]")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("input.wav", recording, fs)
    return "input.wav"

def main():
    print(f"Agent Ready using Groq ({GROQ_MODEL})")
    while True:
        # STEP 1: Local Speech-to-Text
        audio_path = record_audio()
        segments, _ = stt_model.transcribe(audio_path, beam_size=5)
        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"You: {user_text}")
        if "exit" in user_text.lower(): 
            print("Goodbye!")
            break

        # STEP 2: The Brain (Groq API)
        print("Groq is thinking...")
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a concise voice assistant. Give 1-2 sentence answers."},
                    {"role": "user", "content": user_text},
                ],
                model=GROQ_MODEL,
            )
            reply = chat_completion.choices[0].message.content
            print(f"Agent: {reply}")
        except Exception as e:
            print(f"Groq Error: {e}")
            continue

        # STEP 3: Local Text-to-Speech
        print("Speaking...")
        samples, sample_rate = tts.create(reply, voice="af_bella", speed=1.1)
        sd.play(samples, sample_rate)
        sd.wait()

if __name__ == "__main__":
    main()