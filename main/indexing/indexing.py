import os
import re
from dataclasses import dataclass
from typing import Any, List, Tuple
from main.paths import INDEX_PATH
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from main.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    load_db,
    load_index,
    load_model,
)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page_number: int
    source_doc: str
    chunk_type: str = "prose"


def looks_like_toc(text: str) -> bool:
    """Heuristic: table-of-contents entries use dot-leader sequences."""
    return len(re.findall(r"\. \. \. \.", text)) > 3


def split_table_and_prose(text: str) -> List[Tuple[str, str]]:
    lines = text.split("\n")
    blocks: List[Tuple[str, str]] = []
    current_block: List[str] = []
    current_type: str | None = None
    active_header = ""

    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            active_header = stripped

        line_type = "table" if stripped.startswith("|") else "prose"

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


def chunk_large_table(table_text: str, max_rows_per_chunk: int = 15) -> List[str]:
    lines = table_text.strip().split("\n")
    if len(lines) <= 2:
        return [table_text]

    header = lines[:2]
    body = lines[2:]

    return [
        "\n".join(header + body[i : i + max_rows_per_chunk])
        for i in range(0, len(body), max_rows_per_chunk)
    ]


def build_chunks_for_page(
    page_text: str,
    page_number: int,
    source_doc: str,
    chunk_id_start: int,
    text_splitter: RecursiveCharacterTextSplitter,
    large_table_token_threshold: int = 600,
) -> Tuple[List[Chunk], int]:
    all_chunks: List[Chunk] = []
    chunk_id = chunk_id_start

    for block_type, block_text in split_table_and_prose(page_text):
        if not block_text.strip():
            continue

        if block_type == "table":
            if looks_like_toc(block_text):
                continue
            table_pieces = (
                chunk_large_table(block_text)
                if len(block_text) > large_table_token_threshold
                else [block_text]
            )

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

def run_ingestion(file: str) -> None:
    model = load_model()
    index = load_index()
    conn, cur = load_db()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    max_id = cur.execute("SELECT MAX(chunk_id) FROM chunk_records").fetchone()[0]
    chunk_id_counter = (max_id or 0) + 1

    doc = fitz.open(file)
    total_pages = len(doc)

    rows: List[Tuple[int, str, str, int, str, int]] = []
    texts: List[str] = []
    ids: List[int] = []

    print(f"Extracting & Chunking {os.path.basename(file)} ({total_pages} pages)...")
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text("text")

        if page_text.strip():
            page_chunks, chunk_id_counter = build_chunks_for_page(
                page_text=page_text,
                page_number=page_num + 1,
                source_doc=file,
                chunk_id_start=chunk_id_counter,
                text_splitter=text_splitter,
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
                        total_pages,
                    )
                )

    doc.close()

    if not rows:
        return

    cur.executemany(
        "INSERT INTO chunk_records (chunk_id, text, source_doc, page_number, chunk_type, page_count) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    print(f"Embedding {len(texts)} chunks from {os.path.basename(file)}...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    index.add(ids, embeddings)
    index.save(str(INDEX_PATH))
