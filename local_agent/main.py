from ollama import chat
import time

user_query = 'Can you explain prompt engineeing?'

prompt = f'''ROLE: Efficient Logic Assistant.
CONSTRAINTS: 
- Direct answers only.
- No "Sure!" or "I can help with that." 
- No conversational filler.
- If the task is simple, respond in less words according to the query.
- For truely simple tasks such as 'Hi' or 'Tell me something' try to keep it under 10 or 20 or 30 words more only if necessary.
- If user asks something to explain give a good explaination not a big paragraph but you can provide some points along with a that paragraph.
USER_QUERY: {user_query}
'''
start = time.time()
stream = chat(
    model='qwen3.5:0.8b',
    messages=[{'role': 'user', 'content': prompt}],
    options={'temperature': 0},
    think=False,
    stream=True,
)

print("Assistant: ", end='', flush=True)

for chunk in stream:
    content = chunk['message']['content']
    print(content, end='', flush=True)
end = time.time()
print(f'\n\nResponse time - {end - start:.2f} seconds')
