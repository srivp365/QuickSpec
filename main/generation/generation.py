import requests
import json
import os
from openrouter import OpenRouter
from dotenv import load_dotenv


# load .env files
load_dotenv()

def run_generation(chunks, question):
    context = "\n\n".join(chunks)
    full_response = ""
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as open_router:
        res = open_router.chat.send(
                messages=[
                    {"content": f"You are answering questions about this document. Use ONLY the context below. If the answer is not contained in the context, say you don't know. Use the least amount of tokens when possible!\n\nContext: {context}\n", "role": "system"},
                    {"content": f"{question}", "role": "user"},
                ],
                model="openrouter/free",
                stream=True,
            )

        for chunk in res:
             # holy crap, figuring out this one line was genuienly a nightmare,
             # I really wished there was more documentation on what exactly an API responds with 💀
             # more and more docs seem to treat non-js sdks as after thoughts, going all in agents
             # spent almost 2 hours debugging the structure of this
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                full_response += token

    print()
    return full_response
