import os
from pathlib import Path
from typing import Any
import time
import typer

from main.config import (
    TOP_K,
    create_schema,
    delete_bm25_index,
    load_bm25_indexing,
    load_db,
)
from main.display import (
    console,
    make_progress,
    print_error,
    print_success,
    render_answer_panel,
    render_stats_table,
    render_sources_table
)
from main.generation.generation import run_generation
from main.indexing.indexing import run_ingestion
from main.retrieval.retrieval import get_chunks, search

app = typer.Typer(help="QuickSpec: local hybrid RAG over hardware datasheets.")


@app.command()
def index(folder: Path = typer.Argument(..., help="Folder of PDFs to ingest")) -> None:
    """Ingest a folder of datasheets into the local index."""
    import os
    from main.config import delete_bm25_index, load_bm25_indexing
    from main.display import print_success
    from main.indexing.indexing import run_ingestion  # Lazy load heavy PDF/splitters

    if not folder.is_dir():
        print_error(f"{folder} is not a directory")
        raise typer.Exit(code=1)

    create_schema()
    console.print(f"[bold]Indexing datasheets from[/bold] {folder}")

    with os.scandir(folder) as entries:
        pdf_files = [
            os.path.join(folder, entry.name)
            for entry in entries
            if entry.is_file() and entry.name.lower().endswith(".pdf")
        ]

    if not pdf_files:
        print_error(f"No PDF files found in {folder}")
        raise typer.Exit(code=1)

    delete_bm25_index()

    for file in pdf_files:
        run_ingestion(file)

    load_bm25_indexing()
    print_success("Indexing complete.")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    method: str = typer.Option("hybrid_reranked", help="Retrieval method"),
    k: int = typer.Option(5, help="Number of chunks"),
) -> None:
    t0 = time.perf_counter()

    with console.status(f"[cyan]Retrieving ({method})...[/cyan]"):
        retrieved_ids = search(method, question, k=k)
    t_search = time.perf_counter()

    if not retrieved_ids:
        print_error("No relevant chunks found.")
        raise typer.Exit(code=1)

    chunk_texts, source_docs, page_numbers = get_chunks(retrieved_ids)
    t_chunks = time.perf_counter()

    sources = [
        {"text": text, "source_doc": doc, "page_number": page}
        for text, doc, page in zip(chunk_texts, source_docs, page_numbers)
    ]

    console.print(f"\n[bold cyan]Answer:[/bold cyan]\n")
    t_gen_start = time.perf_counter()
    answer = run_generation(chunk_texts, source_docs, page_numbers, question)
    t_gen_end = time.perf_counter()

    render_sources_table(sources)

    console.print(
        f"\n[dim]Timings: Search: {t_search - t0:.2f}s | "
        f"DB Fetch: {t_chunks - t_search:.2f}s | "
        f"LLM Generation: {t_gen_end - t_gen_start:.2f}s | "
        f"Total: {t_gen_end - t0:.2f}s[/dim]"
    )


@app.command()
def chat(
    method: str = typer.Option("hybrid_reranked", help="Retrieval method to use"),
    k: int = typer.Option(TOP_K, help="Number of chunks to retrieve"),
) -> None:
    """Interactive REPL for back-to-back questions."""
    console.print(
        "[bold cyan]QuickSpec chat[/bold cyan] -- type 'exit' or Ctrl+C to quit\n"
    )

    while True:
        try:
            question = console.input("[bold]> [/bold]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.strip().lower() in ("exit", "quit"):
            break
        if not question.strip():
            continue

        retrieved_ids = search(method, question, k=k)
        if not retrieved_ids:
            print_error("No relevant chunks found.")
            continue

        chunk_texts, source_docs, page_numbers = get_chunks(retrieved_ids)

        sources = [
            {"text": text, "source_doc": doc, "page_number": page}
            for text, doc, page in zip(chunk_texts, source_docs, page_numbers)
        ]

        answer = run_generation(chunk_texts, source_docs, page_numbers, question)
        render_answer_panel(question, answer, sources)
        console.print()


@app.command()
def stats() -> None:
    """Show corpus statistics."""
    _, cur = load_db()

    total = cur.execute("SELECT COUNT(*) FROM chunk_records").fetchone()[0]
    by_type_rows = cur.execute(
        "SELECT chunk_type, COUNT(*) FROM chunk_records GROUP BY chunk_type"
    ).fetchall()
    docs = cur.execute("SELECT DISTINCT source_doc FROM chunk_records").fetchall()

    stats_dict: dict[str, Any] = {
        "total_chunks": total,
        "by_type": {row[0]: row[1] for row in by_type_rows},
        "source_docs": [d[0] for d in docs],
    }

    render_stats_table(stats_dict)


if __name__ == "__main__":
    app()
