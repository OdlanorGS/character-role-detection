from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
from anthropic import Anthropic

print("ANTHROPIC_API_KEY loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20,
    messages=[
        {"role": "user", "content": "Reply with exactly: hello"}
    ],
)

print(response.content[0].text)