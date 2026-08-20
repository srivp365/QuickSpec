import math
import re
from typing import List, Tuple, Union

import numpy as np
from usearch.index import Matches

from main.config import (
    TOP_K,
    load_bm25_indexing,
    load_db,
    load_index,
    load_model,
    load_reranker,
)


def _sigmoid(x: float) -> float:
    """Normalize raw cross-encoder logits to [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# In main/retrieval/retrieval.py
def search(method: str, query: str, k: int = TOP_K) -> List[int]:
    if method == "usearch":
        return get_usearch_ids(query, k)
    if method == "bm25":
        return get_bm25_ids(query, k)
    if method == "hybrid":
        usearch_match_ids = get_usearch_ids(query, k)
        bm_25_ids = get_bm25_ids(query, k)
        return reciprocal_rank_fusion(usearch_match_ids, bm_25_ids)[:k]
    if method == "db":
        _, cur = load_db()
        rows = cur.execute("SELECT chunk_id FROM chunk_records").fetchall()
        return [row["chunk_id"] for row in rows]
    if method == "hybrid_reranked":
        # 12 candidates provides high recall while keeping CPU reranking under 1.5s
        candidate_pool_size = max(k * 2, 12)
        usearch_match_ids = get_usearch_ids(query, candidate_pool_size)
        bm_25_ids = get_bm25_ids(query, candidate_pool_size)
        fused = reciprocal_rank_fusion(usearch_match_ids, bm_25_ids)[:candidate_pool_size]
        reranked = rerank(query, fused, top_k=k)
        return [chunk_id for chunk_id, _ in reranked]


def rerank(query: str, candidate_ids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
    if not candidate_ids:
        return []

    reranker = load_reranker()
    _, cur = load_db()

    placeholders = ",".join("?" * len(candidate_ids))
    rows = cur.execute(
        f"SELECT chunk_id, text FROM chunk_records WHERE chunk_id IN ({placeholders})",
        candidate_ids,
    ).fetchall()

    row_map = {row["chunk_id"]: row["text"] for row in rows}
    valid_chunks = [(cid, row_map[cid]) for cid in candidate_ids if cid in row_map]

    if not valid_chunks:
        return []

    # 1000 characters ensures full section headers and leading lists fit in the context
    pairs = [[query, text[:1000]] for _, text in valid_chunks]
    raw_scores = reranker.predict(pairs)

    normalized_scores = [_sigmoid(s) for s in raw_scores]
    ranked = sorted(
        zip(valid_chunks, normalized_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [(chunk_id, float(score)) for (chunk_id, _), score in ranked[:top_k]]

def get_bm25_ids(query: str, k: int) -> List[int]:
    _, cur = load_db()
    bm25_index, bm25_ids = load_bm25_indexing()
    tokenized_query = re.findall(r"\w+", query.lower())

    scores = np.array(bm25_index.get_scores(tokenized_query))
    if len(scores) == 0:
        return []

    # Fast top-K selection without full array sorting
    if len(scores) <= k:
        top_k_indices = np.argsort(-scores)
    else:
        partitioned = np.argpartition(-scores, k)[:k]
        top_k_indices = partitioned[np.argsort(-scores[partitioned])]

    return [bm25_ids[i] for i in top_k_indices if scores[i] > 0]


def get_usearch_ids(query: str, k: int) -> List[int]:
    index = load_index()
    model = load_model()

    instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(f"{instruction}{query}", normalize_embeddings=True)

    matches: Matches = index.search(query_embedding, count=k).to_list()
    return [match[0] for match in matches]


def get_chunks(chunk_ids: List[Union[int, Tuple[int, float]]]) -> Tuple[List[str], List[str], List[int]]:
    """
    Accepts chunk IDs or (id, score) tuples, expands contiguous neighbor chunks (N, N+1)
    from the same source document to bridge cross-page cuts, and returns deduped context blocks.
    """
    if not chunk_ids:
        return [], [], []

    clean_ids: List[int] = [
        item[0] if isinstance(item, (list, tuple)) else item for item in chunk_ids
    ]

    # Include next adjacent chunk ID to heal cross-page cuts
    expanded_ids = set()
    for cid in clean_ids:
        expanded_ids.update([cid, cid + 1])

    _, cur = load_db()
    placeholders = ",".join("?" * len(expanded_ids))
    rows = cur.execute(
        f"SELECT chunk_id, text, source_doc, page_number FROM chunk_records WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id",
        list(expanded_ids),
    ).fetchall()

    row_map = {row["chunk_id"]: row for row in rows}

    chunks: List[str] = []
    source_page: List[str] = []
    page_number: List[int] = []

    for cid in clean_ids:
        if cid in row_map:
            doc = row_map[cid]["source_doc"]
            combined_text = row_map[cid]["text"]

            # Merge neighbor chunk if it belongs to the same document
            if (cid + 1) in row_map and row_map[cid + 1]["source_doc"] == doc:
                combined_text += "\n" + row_map[cid + 1]["text"]

            chunks.append(combined_text)
            source_page.append(doc)
            page_number.append(row_map[cid]["page_number"])

    return chunks, source_page, page_number


def run_retrieval(query: str, k: int = TOP_K) -> Tuple[List[str], List[str], List[int]]:
    """Returns (chunks, source_docs, page_numbers)."""
    chunk_ids = search("hybrid_reranked", query, k=k)
    return get_chunks(chunk_ids)


def reciprocal_rank_fusion(dense_ids: List[int], bm25_ids: List[int], k: int = 60) -> List[int]:
    fused_scores: dict[int, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    scored_tuple_list = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [scored_list[0] for scored_list in scored_tuple_list]
