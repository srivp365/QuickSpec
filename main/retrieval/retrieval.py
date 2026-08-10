from usearch.index import Matches

from main.config import load_bm25_indexing, load_db, load_index, load_model

# steps
# 1. make query
# 2. make vector
# 3. query usearch
# 4. use that to get model to answer questions

query = "What is the best bar for Engineering Students?"


def search(query):
    model = load_model()
    index = load_index()

    # print(f"This is the query: {query}")
    query_embedding = model.encode(query)

    # find vector matches
    matches: Matches = index.search(query_embedding, count=10).to_list()
    match_ids = [match[0] for match in matches]
    # print(f"This is match_ids: {match_ids}")
    return match_ids


def run_retrieval(query):
    _, cur = load_db()
    if isinstance(query, list):
        for item in query:
            row = cur.execute(
                "SELECT * FROM chunk_records WHERE chunk_id = ?", (item,)
            ).fetchone()
            # print(f"This is the text stored in this chunk!: {row['text']} \n \n")
            return row["text"]

    if isinstance(query, str):
        bm25_index, bm25_ids = load_bm25_indexing()
        tokenized_query = query.split(" ")
        bm25_match_ids = bm25.get_scores(tokenized_query)
        ranked = sorted(zip(bm25_chunk_ids, scores), key=lambda x: x[1], reverse=True)
        top_ids = [chunk_id for chunk_id, score in ranked[:k]]
        usearch_match_ids = search(query)
        chunks = []
        source_docs = []
        pages = []
        # find chunks in db
        for id in usearch_match_ids:
            row = cur.execute(
                "SELECT * FROM chunk_records WHERE chunk_id = ?", (id,)
            ).fetchone()
            chunks.append(row["text"])
            source_docs.append(row["source_doc"])
            pages.append(row["page_number"])

        return chunks, source_docs, pages, top_ids


def reciprocal_rank_fusion(dense_ids, bm25_ids, k=60):
    fused_scores = {}
    for rank, chunk_id in enumerate(dense_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
