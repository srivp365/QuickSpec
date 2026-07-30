from sentence_transformers import SentenceTransformer
import sqlite3
from usearch.index import Index


# consts used for model quantization
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 20
RERANK_TOP_K = 5
RRF_K = 60

def load_model():
    MODEL_DIR = "data/model/onnx/"
    # declare the model the model (I quantized it, will optimize on another PR) to create embeddings
    model = SentenceTransformer(
        MODEL_DIR,
        backend="onnx",
        model_kwargs={"file_name": "model_qint8_avx512_vnni.onnx", "subfolder": "onnx"}
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

# code used to make initial index
# index = Index(
#     ndim=384, # Define the number of dimensions in for a single input vectors (i was passing in the shape of the numpy array produced from an embedding 🤦)
#     metric='cos', # Choose 'l2sq', 'ip', 'haversine' or other metric, default = 'cos'
#     dtype='f32', # Quantize to 'f16', 'e5m2', 'e4m3', 'e3m2', 'e2m3', 'u8', 'i8', 'b1'..., default = None
#     connectivity=16, # How frequent should the connections in the graph be, optional
#     expansion_add=128, # Control the recall of indexing, optional
#     expansion_search=64, # Control the quality of search, optional
# )
