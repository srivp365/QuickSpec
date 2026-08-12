# eval set + judge written by Claude Sonnet 5 (verfied by me)

"""
Bare-bones eval harness.

Assumes eval_set.json has been updated so each entry has a
"relevant_chunk_ids" field (list of int) -- map these in yourself
by running ingestion, then checking which chunk_id(s) landed on each
question's "page_number" for its "source_doc".

Wire in your own search(), get_chunks(), and generate() functions
from your retrieval/generation modules.
"""

import json
import os

from dotenv import load_dotenv
from openrouter import OpenRouter


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0



def reciprocal_rank(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1 / (i + 1)
    return 0


def llm_judge(expected_answer, generated_answer, judge_fn):
    """judge_fn: your OpenRouter call, takes (expected, generated) -> bool"""
    prompt = (
        f"Expected answer: {expected_answer}\n"
        f"Generated answer: {generated_answer}\n"
        "Does the generated answer correctly convey the expected answer? "
        "Reply only YES or NO."
    )
    response = judge_fn(prompt)
    return response.strip().upper().startswith("YES")


def run_eval(eval_set_path, search_fn, get_chunks_fn, generate_fn, judge_fn, k=50):
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    p_at_k, mrrs, gen_correct = [], [], []

    for qa in eval_set:
        # print(f"This is the question from inside run_eval {qa['question']}")
        retrieved_ids = search_fn("hybrid_reranked", qa["question"])
        relevant_ids = qa["relevant_chunk_ids"]
        # print(f"this is retrieved: {retrieved_ids} and this is relevant {relevant_ids}")
        recall = recall_at_k(retrieved_ids, relevant_ids)
        p_at_k.append(precision_at_k(retrieved_ids, relevant_ids, k))
        mrrs.append(reciprocal_rank(retrieved_ids, relevant_ids))
        # print(f"This is retrieved chunks!: {retrieved_ids[:k]}")
        chunks, source_docs, pages = get_chunks_fn(retrieved_ids[:k])
        # print(f"This is chunks!: {chunks}")
        answer = generate_fn(chunks, source_docs, pages, qa["question"])
        gen_correct.append(judge_fn(qa["expected_answer"], answer))
        raw_response = judge_fn(qa["expected_answer"], answer)
        print(f"Q: {qa['question'][:50]}\nJUDGE RAW OUTPUT: {raw_response}\n")

    return {
        f"recall@{k}": {recall},
        "mrr": sum(mrrs) / len(mrrs),
        "gen_accuracy": sum(gen_correct) / len(gen_correct),
    }


load_dotenv()


def judge(expected_answer, generated_answer):
    prompt = (
        f"Expected answer: {expected_answer}\n"
        f"Generated answer: {generated_answer}\n"
        "The generated answer may be a short phrase or fragment rather than a full "
        "sentence -- that's fine and expected. Judge only whether it states the same "
        "fact as the expected answer, ignoring differences in phrasing, units notation "
        "(e.g. '27' vs '27 ohms' vs '27Ω' are equivalent), or completeness of explanation. "
        "Reply only YES or NO."
    )

    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as open_router:
        res = open_router.chat.send(
            messages=[{"content": prompt, "role": "user"}],
            model="anthropic/claude-haiku-4.5",
            stream=False,
        )
        response_text = res.choices[0].message.content

    print(
        f"RAW MODEL TEXT: {response_text!r}"
    )
    return str(response_text).strip().upper().startswith("YES")


if __name__ == "__main__":
    from main.generation.generation import run_generation
    from main.retrieval.retrieval import get_chunks, search

    results = run_eval("eval_set.json", search, get_chunks, run_generation, judge)
    print(results)
