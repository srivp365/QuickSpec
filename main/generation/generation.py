import os
from typing import List

from dotenv import load_dotenv
from openrouter import OpenRouter

# load .env files
load_dotenv()


# In generation.py
def run_generation(chunks: List[str], source_docs: List[str], pages: List[int], question: str) -> str:
    context_blocks = [
        f"--- Source: {source_docs[i]} | Page {pages[i]} ---\n{chunk}"
        for i, chunk in enumerate(chunks)
    ]
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert technical documentation assistant. "
        "Your goal is to answer accurately, referencing only the provided context.\n\n"
        f"Context:\n{context}\n"
    )

    full_response = ""
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as open_router:
        res = open_router.chat.send(
            messages=[
                {"content": system_prompt, "role": "system"},
                {"content": question, "role": "user"},
            ],
            model="google/gemini-2.5-flash-lite",
            stream=True,
        )

        for chunk in res:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                full_response += token

    print()
    return full_response
