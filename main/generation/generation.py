import requests
import json
import os
from openrouter import OpenRouter
from dotenv import load_dotenv


# load .env files
load_dotenv()

def run_generation(chunks, source_docs, pages, question):
    context_blocks = []
    for i, chunk in enumerate(chunks):
        header = f"--- Source: {source_docs[i]} | Page {pages[i]} ---"
        context_blocks.append(f"{header}\n{chunk}")

    context = "\n\n".join(context_blocks)
    full_response = ""
    system_prompt = ("You are an expert embedded systems engineer answering questions based ONLY on the context below. "
            "If the answer is not contained in the context, say 'I don't know'.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Preserve exact technical specifications: If the text includes quantities, architectural layouts (e.g., 'banks', 'channels'), or specific dimensions, you MUST include them.\n"
            "2. Preserve exact terminology: Never substitute specific pin names, voltage domains (e.g., 'IOVDD'), or package specs with generic terms (like 'tied high' or 'power supply').\n"
            "3. Be concise, but NEVER sacrifice technical completeness for brevity.\n\n"
            f"Context:\n{context}\n")
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as open_router:
        res = open_router.chat.send(
                messages=[
                    {"content": f"{system_prompt}", "role": "system"},
                    {"content": f"{question}", "role": "user"},
                ],
                model="google/gemini-2.5-flash-lite",
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
