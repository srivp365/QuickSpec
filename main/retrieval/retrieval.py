from usearch.index import Matches

from main.config import (
    load_bm25_indexing,
    load_db,
    load_index,
    load_model,
    load_reranker,
)

# steps
# 1. make query
# 2. make vector
# 3. query usearch
# 4. use that to get model to answer questions

query = "What is the best bar for Engineering Students?"


def search(method, query, k=20):
    if method == "usearch":
        return get_usearch_ids(query, k)
    if method == "bm25":
        return get_bm25_ids(query, k)
    if method == "hybrid":
        usearch_match_ids = get_usearch_ids(query)
        bm_25_ids = get_bm25_ids(query, k)
        return reciprocal_rank_fusion(usearch_match_ids, bm_25_ids)[:k]
    if method == "db":
        return get_db_entries(query)
    if method == "hybrid_reranked":
        usearch_match_ids = get_usearch_ids(query)
        bm_25_ids = get_bm25_ids(query, k)
        fused = reciprocal_rank_fusion(usearch_match_ids, bm_25_ids)[
            :k
        ]  # top 20 candidates
        return rerank(query, fused, top_k=5)  # rescored down to top 5


def rerank(query, candidate_ids, top_k=5):
    reranker = load_reranker()
    _, cur = load_db()
    chunks = []
    for chunk_id in candidate_ids:
        row = cur.execute(
            "SELECT * FROM chunk_records WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        chunks.append((chunk_id, row["text"]))

    pairs = [[query, text] for chunk_id, text in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for (chunk_id, text), score in ranked[:top_k]]


def get_bm25_ids(query, k):
    _, cur = load_db()
    # building the bm_25 index
    bm25_index, bm25_ids = load_bm25_indexing(cur)
    tokenized_query = query.split(" ")
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(zip(bm25_ids, scores), key=lambda x: x[1], reverse=True)
    top_ids = [chunk_id for chunk_id, score in ranked[:k]]
    return top_ids


def get_usearch_ids(query):
    index = load_index()
    model = load_model()

    query_embedding = model.encode(query)

    # find vector matches
    matches: Matches = index.search(query_embedding, count=10).to_list()
    match_ids = [match[0] for match in matches]
    # print(f"This is match_ids: {match_ids}")
    return match_ids


def get_chunks(chunk_ids: list[int]):
    _, cur = load_db()
    chunks = []
    for chunk_id in chunk_ids:
        row = cur.execute(
            "SELECT * FROM chunk_records WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row:
            chunks.append(row["text"])
    return chunks


def run_retrieval(query):
    _, cur = load_db()
    match_ids = search("hybrid", query)
    chunks = []
    source_docs = []
    pages = []
    for id in match_ids:
        row = cur.execute(
            "SELECT * FROM chunk_records WHERE chunk_id = ?", (id,)
        ).fetchone()
        chunks.append(row["text"])
        source_docs.append(row["source_doc"])
        pages.append(row["page_number"])

    return chunks, source_docs, pages


def reciprocal_rank_fusion(dense_ids, bm25_ids, k=60):
    fused_scores = {}
    for rank, chunk_id in enumerate(dense_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    scored_tuple_list = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    res = [scored_list[0] for scored_list in scored_tuple_list]
    return res
