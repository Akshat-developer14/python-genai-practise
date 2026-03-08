# Prompt style

## Alpaca Prompt
### Instructions: <SYSTEM_PROMPT>\n 
### Input: <user_query>
### Response: \n

## ChatML Prompt
{
    "role": "system" | "user" | "assistant", 
    "content": "string"
}

## INST Prompting (Instruction prompting) - used by llama2
[INST] What is the time now? [/INST]