from main.retrieval.retrieval import search
from main.config import load_db
import sys
from main.retrieval.retrieval import get_usearch_ids, get_bm25_ids, reciprocal_rank_fusion
from main.config import load_reranker

# The 5 questions with recall=0.0 despite ground truth existing in the DB
FAILING_QUESTIONS = {
    3: ("How many multifunction GPIO pins does the RP2040 have?", [45]),
    4: ("How many dedicated IO pins does the RP2040 have for SPI Flash?", [45]),
    6: ("What USB standard does the RP2040 support?", [45]),
    9: ("What happens to the RUN pin if no external reset is required?", [47]),
    12: ("What does XIN also function as if XOUT is disconnected?", [47]),
}


def preview_chunk(cur, chunk_id, chars=200):
    row = cur.execute(
        "SELECT text FROM chunk_records WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return row["text"][:chars].replace("\n", " ") if row else "MISSING FROM DB"

def check_chunk():
    _, cur = load_db()
    row = cur.execute("SELECT text FROM chunk_records WHERE chunk_id = ?", (45,)).fetchone()
    print(len(row["text"]))
    print(row["text"][:1000])

def query_check():
    query = "How many multifunction GPIO pins does the RP2040 have?"
    usearch_ids = get_usearch_ids(query, 20)
    bm25_ids = get_bm25_ids(query, 20)
    fused = reciprocal_rank_fusion(usearch_ids, bm25_ids)[:20]

    _, cur = load_db()
    reranker = load_reranker()
    pairs = []
    for cid in fused:
        row = cur.execute("SELECT text FROM chunk_records WHERE chunk_id = ?", (cid,)).fetchone()
        pairs.append([query, row["text"][:1000]])

    scores = reranker.predict(pairs)
    for cid, score in sorted(zip(fused, scores), key=lambda x: x[1], reverse=True):
        marker = " <-- EXPECTED" if cid == 45 else ""
        print(f"{cid}: {score:.4f}{marker}")


def diagnose():
    _, cur = load_db()

    for idx, (question, expected) in FAILING_QUESTIONS.items():
        print(f"\n{'='*70}\nQ{idx}: {question}\nExpected: {expected}\n{'='*70}")

        # Check each stage of the pipeline separately
        usearch_ids = search("usearch", question, k=10)
        bm25_ids = search("bm25", question, k=10)
        hybrid_ids = search("hybrid", question, k=20)
        reranked_ids = search("hybrid_reranked", question, k=5)

        for label, ids in [
            ("usearch (dense)", usearch_ids),
            ("bm25 (sparse)", bm25_ids),
            ("hybrid (fused)", hybrid_ids),
            ("hybrid_reranked (final)", reranked_ids),
        ]:
            hit = bool(set(ids) & set(expected))
            marker = "HIT" if hit else "miss"
            print(f"\n  [{marker}] {label}: {ids}")

        print(f"\n  Expected chunk preview(s):")
        for cid in expected:
            print(f"    {cid}: {preview_chunk(cur, cid)}")


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "diagnose":
        diagnose()

    if command == "check_chunk":
        check_chunk()

    if command == "query_check":
        query_check()
