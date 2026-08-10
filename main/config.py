import sqlite3

import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from usearch.index import Index

# consts used for model quantization
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 10
RERANK_TOP_K = 5
RRF_K = 60


def load_model():
    MODEL_DIR = "data/model/onnx/"
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = (
        4  # leave headroom, adjust based on your core count
    )
    session_options.inter_op_num_threads = 1

    # declare the model the model (I quantized it, will optimize on another PR) to create embeddings
    model = SentenceTransformer(
        MODEL_DIR,
        backend="onnx",
        model_kwargs={
            "file_name": "model_qint8_avx512_vnni.onnx",
            "subfolder": "onnx",
            "provider_options": None,
            "session_options": session_options,
        },
    )

    return model


def load_index():
    index = Index.restore("data/db/index.usearch")
    return index


def load_db():
    conn = sqlite3.connect("data/db/chunks.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def load_bm25_indexing(cur):
    rows = cur.execute("SELECT chunk_id, text FROM chunk_records").fetchall()
    ids = [r[0] for r in rows]
    tokenized_corpus = [r[1].lower().split(" ") for r in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, ids


# code used to make initial index
# index = Index(
#     ndim=384,  # Define the number of dimensions in for a single input vectors (i was passing in the shape of the numpy array produced from an embedding 🤦)
#     metric="cos",  # Choose 'l2sq', 'ip', 'haversine' or other metric, default = 'cos'
#     dtype="f32",  # Quantize to 'f16', 'e5m2', 'e4m3', 'e3m2', 'e2m3', 'u8', 'i8', 'b1'..., default = None
#     connectivity=16,  # How frequent should the connections in the graph be, optional
#     expansion_add=128,  # Control the recall of indexing, optional
#     expansion_search=64,  # Control the quality of search, optional
# )
