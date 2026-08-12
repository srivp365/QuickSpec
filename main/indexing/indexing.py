import os
from dataclasses import dataclass
import re
import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from main.config import load_bm25_indexing, load_db, load_index, load_model


@dataclass
class Page_Chunk:
    text: str
    source_doc: str
    page_number: int
    page_count: int
    title: str = ""
    author: str = ""

    @classmethod
    def from_pumupdf_to_page(cls, raw: dict) -> Page_Chunk:
        metadata = raw["metadata"]
        return cls(
            text=raw["text"],
            source_doc=metadata["file_path"],
            page_number=metadata["page_number"],
            page_count=metadata["page_count"],
            title=metadata.get("title", ""),
            author=metadata.get("author", ""),
        )


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page_number: int
    source_doc: str
    chunk_type: str = "prose"


model = load_model()
index = load_index()
conn, cur = load_db()


def looks_like_toc(text: str) -> bool:
    """Heuristic: table-of-contents entries use dot-leader sequences
    (". . . . .") to connect a heading to its page number."""
    dot_leader_count = len(re.findall(r"\. \. \. \.", text))
    return dot_leader_count > 3  # tune threshold based on false positive/negative rate


def split_table_and_prose(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    blocks = []
    current_block = []
    current_type = None
    active_header = ""

    for line in lines:
        if re.match(r"^#{1,6}\s+", line.strip()):
            active_header = line.strip()

        is_table_line = line.strip().startswith("|")
        line_type = "table" if is_table_line else "prose"

        if current_type is None:
            current_type = line_type

        if line_type != current_type:
            block_content = "\n".join(current_block)
            if current_type == "table" and active_header:
                block_content = f"{active_header}\n{block_content}"
            blocks.append((current_type, block_content))
            current_block = []
            current_type = line_type

        current_block.append(line)

    if current_block:
        block_content = "\n".join(current_block)
        if current_type == "table" and active_header:
            block_content = f"{active_header}\n{block_content}"
        blocks.append((current_type, block_content))

    return blocks

def chunk_large_table(table_text: str, max_rows_per_chunk: int = 15) -> list[str]:
    """Splits an oversized table into row-group chunks, repeating the
    header row (and separator row) in every resulting piece."""
    lines = table_text.strip().split("\n")
    if len(lines) <= 2:
        return [table_text]

    header = lines[:2]  # header row + markdown separator row (|---|---|)
    body = lines[2:]

    chunks = []
    for i in range(0, len(body), max_rows_per_chunk):
        chunk_lines = header + body[i : i + max_rows_per_chunk]
        chunks.append("\n".join(chunk_lines))
    return chunks


def build_chunks_for_page(
    page_text: str,
    page_number: int,
    source_doc: str,
    chunk_id_start: int,
    text_splitter,
    Chunk,
    large_table_token_threshold: int = 600,
) -> tuple[list, int]:
    """Runs table-aware chunking on one page's text. Returns (chunks, next_chunk_id).
    Table blocks are kept whole (or split by row-group if oversized);
    prose blocks go through the normal text_splitter unchanged."""
    all_chunks = []
    chunk_id = chunk_id_start

    for block_type, block_text in split_table_and_prose(page_text):
        if not block_text.strip():
            continue

        if block_type == "table":
            if looks_like_toc(block_text):
                continue
            if len(block_text) > large_table_token_threshold:
                table_pieces = chunk_large_table(block_text)
            else:
                table_pieces = [block_text]

            for piece in table_pieces:
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=piece,
                        page_number=page_number,
                        source_doc=source_doc,
                        chunk_type="table",
                    )
                )
                chunk_id += 1
        else:
            for split_text in text_splitter.split_text(block_text):
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=split_text,
                        page_number=page_number,
                        source_doc=source_doc,
                        chunk_type="prose",
                    )
                )
                chunk_id += 1

    return all_chunks, chunk_id


def get_files(file_path):
    # get all files from a folder and run ingestion on each one before adding it to a db
    with os.scandir(file_path) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
    for f in files:
        print(f"\n \n This is the file: {f} \n \n")
        file = f"{file_path}/{f}"
        run_ingestion(model, index, conn, cur, file)
    load_bm25_indexing(cur)


def run_ingestion(model, index, conn, cur, file):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=800, chunk_overlap=100
    )

    x = cur.execute("SELECT MAX(chunk_id) FROM chunk_records ").fetchone()[0]
    if x is None:
        x = 0

    rows = []
    texts = []
    ids = []
    chunk_id_counter = x + 1

    doc = fitz.open(file)
    num_pages = len(doc)
    doc.close()

    raw_pages = []
    for page_num in tqdm(range(num_pages), desc=f"Extracting {os.path.basename(file)}"):
        page_data = pymupdf4llm.to_markdown(
            file, pages=[page_num], page_chunks=True, use_ocr=False
        )
        raw_pages.extend(page_data)

    for raw_page in raw_pages:
        ps = Page_Chunk.from_pumupdf_to_page(raw_page)

        page_chunks, chunk_id_counter = build_chunks_for_page(
            page_text=ps.text,
            page_number=ps.page_number,
            source_doc=ps.source_doc,
            chunk_id_start=chunk_id_counter,
            text_splitter=text_splitter,
            Chunk=Chunk,
        )

        for c in page_chunks:
            texts.append(c.text)
            ids.append(c.chunk_id)
            rows.append(
                (
                    c.chunk_id,
                    c.text,
                    c.source_doc,
                    c.page_number,
                    c.chunk_type,
                    ps.page_count,
                )
            )

    cur.executemany(
        "INSERT INTO chunk_records (chunk_id, text, source_doc, page_number, chunk_type, page_count) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    index.add(ids, embeddings)
    index.save("data/db/index.usearch")
