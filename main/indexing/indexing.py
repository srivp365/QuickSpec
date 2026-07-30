from dataclasses import dataclass
import numpy as np
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from main.config import load_model, load_db, load_index
import os

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



model = load_model()
index = load_index()
conn, cur = load_db()


def get_files(file_path):
    # get all files from a folder and run ingestion on each one before adding it to a db
    with os.scandir(file_path) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
    for f in files:
        file = f"{file_path}/{f}"
        run_ingestion(model, index, conn, cur, file)



def run_ingestion(model, index, conn, cur, file):
    # define the text splitter / basic chunking model
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=800,
        chunk_overlap=100
    )

    # get max number to define consecutive chunk_id
    x = cur.execute("SELECT MAX(chunk_id) FROM chunk_records ").fetchone()[0]

    if x is None:
        x = 0

    rows = []
    texts = []
    ids = []
    chunk_id_counter = x + 1

    # first off, get text from the pdf in md format, sample datasheets + pdfs in assetes
    for raw_page in pymupdf4llm.to_markdown(file, page_chunks=True):
        ps = Page_Chunk.from_pumupdf_to_page(raw_page)
        split_texts = text_splitter.split_text(ps.text)
        for split_text in split_texts:
            texts.append(split_text)
            ids.append(chunk_id_counter)
            rows.append(
                (
                    chunk_id_counter,
                    split_text,
                    ps.source_doc,
                    ps.page_number,
                    "prose",
                    ps.page_count
                )
            )
            chunk_id_counter += 1

    cur.executemany(
        "INSERT INTO chunk_records (chunk_id, text, source_doc, page_number, chunk_type, page_count) VALUES (?,?,?,?,?,?)",
        rows
    )
    conn.commit()


    # add vector embeddings to usearch by batch
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    index.add(ids, embeddings)

    index.save('data/db/index.usearch')


if __name__ == "__main__":
    run_ingestion("data/pdfs/Western_ENG_FROSH_Guide-1-5.pdf")
