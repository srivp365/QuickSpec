from pathlib import Path
from platformdirs import user_data_dir

APP_DATA = Path(user_data_dir("quickspec", "yourname"))

DB_PATH = APP_DATA / "db" / "chunks.db"
INDEX_PATH = APP_DATA / "db" / "index.usearch"
MODEL_PATH = APP_DATA / "model" / "onnx"
CONFIG_PATH = APP_DATA / "config.toml"
RERANKER_PATH = APP_DATA / "model" / "reranker_onnx"
BM25_PATH = APP_DATA / "db" / "bm25.pkl"
EVAL_SET_PATH = APP_DATA / "db" / "eval_set.json"

def ensure_app_dirs():
    (APP_DATA / "db").mkdir(parents=True, exist_ok=True)
    (APP_DATA / "model" / "onnx").mkdir(parents=True, exist_ok=True)
