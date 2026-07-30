from dataclasses import dataclass
import numpy as np
import pymupdf4llm
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from usearch.index import Index

MODEL_DIR = "data/model/onnx/"

@dataclass
class Page_Chunk:
    text : str
    source_doc : str
    page_number : int
    page_count : int
    title : str = ""
    author : str = ""

    @classmethod
    def from_pumupdf_to_page(cls, raw : dict) -> Page_Chunk:
        metadata = raw["metadata"]
        return cls(
            text = raw["text"],
            source_doc = metadata["file_path"],
            page_number = metadata["page_number"],
            page_count = metadata["page_count"],
            title = metadata.get("title", ""),
            author = metadata.get("author", "")

        )

@dataclass
class Chunk:
    chunk_id: int
    text: str
    page_number: int
    source_doc: str
    chunk_type: str = "prose"




# first off, get text from the pdf in md format, sample datasheets + pdfs in assetes
pages = pymupdf4llm.to_markdown("data/pdfs/Western_ENG_FROSH_Guide-1-5.pdf", page_chunks = True)
page_splits = [Page_Chunk.from_pumupdf_to_page(page) for page in pages]

# next, declare the text splitter to create chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=50
)



# establish connection to db
conn = sqlite3.connect("data/db/chunks.db")
cur = conn.cursor()

x = cur.execute("SELECT MAX(chunk_id) FROM chunk_records ").fetchone()[0]

if x is None:
    x = 0

all_chunks : [Chunk] = []
chunk_id_counter = x + 1

for ps in page_splits:
    split_texts = text_splitter.split_text(ps.text)
    for split_text in split_texts:
        all_chunks.append(
            Chunk(
                chunk_id=chunk_id_counter,
                text=split_text,
                page_number=ps.page_number,
                source_doc=ps.source_doc,
            )
        )
        cur.execute(
            "INSERT INTO chunk_records (chunk_id, text, source_doc, page_number, chunk_type, page_count) VALUES (?,?,?,?,?,?)",
            (chunk_id_counter, split_text, ps.source_doc, ps.page_number, "prose", ps.page_count)
        )
        chunk_id_counter += 1
    conn.commit()




# use the model (I quantized it, will optimize on another PR) to create embeddings
model = SentenceTransformer(
    MODEL_DIR,
    backend="onnx",
    model_kwargs={"file_name": "model_qint8_avx512_vnni.onnx", "subfolder": "onnx"}
)



# pushing to usearch
index = Index(
    ndim=384, # Define the number of dimensions in for a single input vectors (i was passing in the shape of the numpy array produced from an embedding 🤦)
    metric='cos', # Choose 'l2sq', 'ip', 'haversine' or other metric, default = 'cos'
    dtype='f32', # Quantize to 'f16', 'e5m2', 'e4m3', 'e3m2', 'e2m3', 'u8', 'i8', 'b1'..., default = None
    connectivity=16, # How frequent should the connections in the graph be, optional
    expansion_add=128, # Control the recall of indexing, optional
    expansion_search=64, # Control the quality of search, optional
)



# add vector embeddings to usearch by batch
texts = [chunk.text for chunk in all_chunks]
ids = [chunk.chunk_id for chunk in all_chunks]
embeddings = model.encode(texts)
index.add(ids, embeddings)

index.save('data/db/index.usearch')
