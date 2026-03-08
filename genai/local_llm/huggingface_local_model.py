from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="google/gemma-3-1b-it")

message = [
    {
        "role": "user",
        "content": "Hey there can you tell me something about yourself like what is your name and job."
    }
]

response = pipe(message)
print(response)