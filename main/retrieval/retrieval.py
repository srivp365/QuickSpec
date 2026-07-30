from usearch.index import Matches
from sentence_transformers import SentenceTransformer
from main.config import load_model, load_index, load_db

# steps
# 1. make query
# 2. make vector
# 3. query usearch
# 4. use that to get model to answer questions

query = "What is the best bar for Engineering Students?"

def run_retrieval(query):
    model = load_model()
    index = load_index()
    conn, cur = load_db()

    # hardcode query
    query_embedding = model.encode(query)

    # find vector matches
    matches : Matches = index.search(query_embedding, count=10).to_list()
    match_ids = [match[0] for match in matches]

    chunks = []
    # find chunks in db
    for id in match_ids:
        row = cur.execute("SELECT * FROM chunk_records WHERE chunk_id = ?", (id, )).fetchone()
        chunks.append(row["text"])


    return chunks
