import os
import sqlite3
import re
import pickle
from functools import lru_cache
from typing import Tuple, Any

# Fast metadata / constants at top level
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
RERANK_TOP_K = 10
RRF_K = 60


def create_schema() -> None:
    conn, cur = load_db()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_records (
            chunk_id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            source_doc TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            page_count INTEGER NOT NULL
        )
    """)
    conn.commit()


class ONNXCrossEncoder:
    def __init__(self, model_dir: str, file_name: str) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSequenceClassification

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            file_name=file_name,
            session_options=session_options,
            provider="CPUExecutionProvider",
        )

    def predict(self, pairs: list[list[str]], batch_size: int = 16) -> Any:
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="np",
        )
        outputs = self.model(**inputs)
        return outputs.logits.flatten()


@lru_cache(maxsize=1)
def load_reranker() -> ONNXCrossEncoder:
    return ONNXCrossEncoder(
        model_dir="data/model/reranker_onnx",
        file_name="model_quantized.onnx"
    )


@lru_cache(maxsize=1)
def load_model():
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer

    MODEL_DIR = "data/model/onnx/"
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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


@lru_cache(maxsize=1)
def load_index(path: str = "data/db/index.usearch"):
    from usearch.index import Index
    if os.path.exists(path):
        return Index.restore(path)
    return Index(
        ndim=384,
        metric="cos",
        dtype="f32",
        connectivity=16,
        expansion_add=128,
        expansion_search=64,
    )


@lru_cache(maxsize=1)
def load_db() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect("data/db/chunks.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA cache_size = -64000;")
    return conn, cur


def build_bm25_from_db(cur: sqlite3.Cursor):
    from rank_bm25 import BM25Okapi
    rows = cur.execute("SELECT chunk_id, text FROM chunk_records ORDER BY chunk_id").fetchall()
    ids = [r[0] for r in rows]
    tokenized_corpus = [re.findall(r"\w+", r[1].lower()) for r in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, ids


@lru_cache(maxsize=1)
def load_bm25_indexing(_cache_key: Any = None, path: str = "data/db/bm25.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["index"], data["ids"]

    _, cur = load_db()
    bm25_index, bm25_ids = build_bm25_from_db(cur)
    save_bm25_index(bm25_index, bm25_ids, path)
    return bm25_index, bm25_ids


def save_bm25_index(bm25_index, bm25_ids: list[int], path: str = "data/db/bm25.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump({"index": bm25_index, "ids": bm25_ids}, f)


def delete_bm25_index(path: str = "data/db/bm25.pkl") -> None:
    if os.path.exists(path):
        os.remove(path)
    load_bm25_indexing.cache_clear()
