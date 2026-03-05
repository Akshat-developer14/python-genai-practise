import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey there! My name is Akshat Sharma and i am a gen ai programmer"
token = enc.encode(text)

print("Tokens: ", token)

decoded = enc.decode(token)
print("Decoded token: ", decoded)