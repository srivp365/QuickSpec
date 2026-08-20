"""
Advanced Evaluation Harness.

Analyzes RAG performance using:
1. Retrieval Metrics: Precision@K, Recall@K, MRR (Mean Reciprocal Rank).
2. Generative Quality: LLM-based semantic faithfulness judge.
3. Reporting: Breakdown of per-question performance.
"""

import json
import os
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()


class RAGEvaluator:
    def __init__(self, model_name: str = "anthropic/claude-haiku-4.5") -> None:
        self.model_name = model_name
        self.client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

    def _llm_judge(
        self, expected: str, generated: str, max_retries: int = 5
    ) -> Tuple[bool, str]:
        prompt = (
            "You are an expert evaluator. Compare the 'Generated Answer' against the 'Expected Answer'.\n"
            "Rules:\n"
            "- Ignore minor phrasing differences.\n"
            "- Accept equivalent notations (e.g., '10V' == '10 volts').\n"
            "- If the generated answer contains correct info but adds fluff, rate as YES.\n"
            "- If the generated answer contradicts or is missing critical specs (e.g., wrong voltage, wrong pin), rate as NO.\n\n"
            f"Expected Answer: {expected}\n"
            f"Generated Answer: {generated}\n\n"
            "Reply with 'YES' or 'NO' followed by a short reason."
        )

        for attempt in range(max_retries):
            try:
                with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
                    res = client.chat.send(
                        messages=[{"content": prompt, "role": "user"}],
                        model=self.model_name,
                        stream=False,
                    )
                    response = res.choices[0].message.content.strip()
                    decision = response.upper().startswith("YES")
                    return decision, response
            except Exception as e:
                wait = 2**attempt
                print(
                    f"Rate limited or error ({e}), retrying in {wait}s... ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(f"Judge call failed after {max_retries} retries")

    def evaluate(
        self,
        eval_set_path: str,
        search_fn: Callable[[str, str], List[int]],
        get_chunks_fn: Callable[[List[int]], Tuple[List[str], List[str], List[int]]],
        generate_fn: Callable[[List[str], List[str], List[int], str], str],
        k_retrieval: int = 10,
        delay_between_questions: float = 1.0,
    ) -> Dict[str, Any]:
        with open(eval_set_path) as f:
            eval_set = json.load(f)

        results: List[Dict[str, Any]] = []
        for qa in eval_set:
            q = qa["question"]
            # Support both 'chunk_ids' and 'relevant_chunk_ids' keys
            relevant = set(qa.get("relevant_chunk_ids", qa.get("chunk_ids", [])))

            # Retrieval
            retrieved = search_fn("hybrid_reranked", q)
            top_k = retrieved[:k_retrieval]

            hits = len(set(top_k) & relevant)
            recall = hits / len(relevant) if relevant else 0
            precision = hits / k_retrieval
            mrr = 0
            for i, rid in enumerate(top_k):
                if rid in relevant:
                    mrr = 1 / (i + 1)
                    break

            # Generation
            chunks, sources, pages = get_chunks_fn(top_k)
            answer = generate_fn(chunks, sources, pages, q)
            is_correct, reason = self._llm_judge(qa["expected_answer"], answer)

            results.append(
                {
                    "question": q,
                    "recall": recall,
                    "precision": precision,
                    "mrr": mrr,
                    "gen_correct": is_correct,
                    "reason": reason,
                }
            )
            print(f"Evaluated: {q[:50]}... -> Correct: {is_correct}")

            time.sleep(delay_between_questions)

        return self._summarize(results)

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        mrrs = [r["mrr"] for r in results]
        accs = [r["gen_correct"] for r in results]

        return {
            "avg_recall": statistics.mean(recalls),
            "avg_precision": statistics.mean(precisions),
            "mrr": statistics.mean(mrrs),
            "generative_accuracy": statistics.mean(accs),
            "details": results,
        }


if __name__ == "__main__":
    from main.generation.generation import run_generation
    from main.retrieval.retrieval import get_chunks, search

    evaluator = RAGEvaluator()
    summary = evaluator.evaluate("eval_set.json", search, get_chunks, run_generation)

    print("\n--- Evaluation Summary ---")
    print(json.dumps(summary, indent=2))