# frontend TUI design done by claude sonnet 5, all underlying logic written by me

import typer
from pathlib import Path

from main.display import (
    render_answer_panel,
    render_eval_table,
    render_stats_table,
    make_progress,
    print_error,
    print_success,
    console,
)

# --- Adjust these imports to match your actual module structure ---
from main.retrieval.retrieval import search, get_chunks
from main.generation.generation import run_generation
from main.config import load_db  # or wherever these live
from main.indexing.indexing import get_files  # your existing ingestion entrypoint
from eval.judge import run_eval, judge  # your existing eval harness

app = typer.Typer(help="QuickSpec: local hybrid RAG over hardware datasheets.")


@app.command()
def index(
    folder: Path = typer.Argument(..., help="Folder of PDFs to ingest"),
):
    """Ingest a folder of datasheets into the local index."""
    if not folder.is_dir():
        print_error(f"{folder} is not a directory")
        raise typer.Exit(code=1)

    create_schema()
    console.print(f"[bold]Indexing datasheets from[/bold] {folder}")

    # get_files already handles per-file ingestion internally; if you want a
    # rich progress bar here, adapt get_files to accept a progress callback,
    # or iterate files here directly instead of calling get_files as a whole.
    get_files(str(folder))

    print_success("Indexing complete.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    method: str = typer.Option("hybrid_reranked", help="Retrieval method: vector, bm25, hybrid, hybrid_reranked"),
    k: int = typer.Option(5, help="Number of chunks to retrieve"),
):
    """Ask a single question against the indexed datasheets."""
    with console.status(f"[cyan]Retrieving ({method})...[/cyan]"):
        retrieved_ids = search(method, question, k=k)

    if not retrieved_ids:
        print_error("No relevant chunks found.")
        raise typer.Exit(code=1)

    _, cur = load_db()
    sources = []
    for chunk_id in retrieved_ids:
        row = cur.execute(
            "SELECT text, source_doc, page_number FROM chunk_records WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row:
            sources.append({"text": row["text"], "source_doc": row["source_doc"], "page_number": row["page_number"]})

    chunk_texts = [s["text"] for s in sources]
    source_docs = [s["source_doc"] for s in sources]
    page_number = [s["page_number"] for s in sources]

    with console.status("[cyan]Generating answer...[/cyan]"):
        answer = run_generation(chunk_texts, source_docs, page_number, question)

    render_answer_panel(question, answer, sources)


@app.command()
def chat(
    method: str = typer.Option("hybrid_reranked", help="Retrieval method to use"),
):
    """Interactive REPL for back-to-back questions."""
    console.print("[bold cyan]QuickSpec chat[/bold cyan] -- type 'exit' or Ctrl+C to quit\n")

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

        retrieved_ids = search(method, question, k=5)
        if not retrieved_ids:
            print_error("No relevant chunks found.")
            continue

        _, cur = load_db()
        sources = []
        for chunk_id in retrieved_ids:
            row = cur.execute(
                "SELECT text, source_doc, page_number FROM chunk_records WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                sources.append({"text": row["text"], "source_doc": row["source_doc"], "page_number": row["page_number"]})

        chunk_texts = [s["text"] for s in sources]
        source_docs = [s["source_doc"] for s in sources]
        page_number = [s["page_number"] for s in sources]
        answer = run_generation(chunk_texts, source_docs, page_number, question)
        render_answer_panel(question, answer, sources)
        console.print()


@app.command()
def eval(
    eval_set_path: Path = typer.Option("eval_set.json", help="Path to eval set JSON"),
    configs: str = typer.Option("usearch,hybrid,hybrid_reranked", help="Comma-separated configs to run"),
):
    """Run the eval harness across one or more retrieval configs and print a comparison table."""
    config_list = [c.strip() for c in configs.split(",")]
    results_by_config = {}

    with make_progress() as progress:
        task = progress.add_task("Running eval configs...", total=len(config_list))
        for config_name in config_list:
            # adjust: your run_eval signature takes search_fn, get_chunks_fn, generate_fn, judge_fn
            # here we bind `method` via a small wrapper so run_eval can stay method-agnostic
            def bound_search(query, _method=config_name):
                return search(query, _method)

            results = run_eval(str(eval_set_path), bound_search, get_chunks, run_generation, judge)
            results_by_config[config_name] = results
            progress.update(task, advance=1)

    render_eval_table(results_by_config)


@app.command()
def stats():
    """Show corpus statistics."""
    _, cur = load_db()

    total = cur.execute("SELECT COUNT(*) FROM chunk_records").fetchone()[0]
    by_type_rows = cur.execute(
        "SELECT chunk_type, COUNT(*) FROM chunk_records GROUP BY chunk_type"
    ).fetchall()
    docs = cur.execute("SELECT DISTINCT source_doc FROM chunk_records").fetchall()

    stats_dict = {
        "total_chunks": total,
        "by_type": {row[0]: row[1] for row in by_type_rows},
        "source_docs": [d[0] for d in docs],
    }

    render_stats_table(stats_dict)


if __name__ == "__main__":
    app()

# import sys
# import time

# import typer

# from main.generation.generation import run_generation
# from main.indexing.indexing import get_files
# from main.retrieval.retrieval import run_retrieval

# app = typer.Typer()


# @app.command()
# def load_documents(file_path):
#     t0 = time.perf_counter()
#     get_files(file_path)
#     t1 = time.perf_counter()

#     print("\n--- Timing ---")
#     print(f"Indexing: {t1 - t0:.3f}s")
#     sys.exit(0)


# @app.command()
# def ask(question):
#     t0 = time.perf_counter()
#     chunks, source_docs, pages = run_retrieval(question)
#     t1 = time.perf_counter()
#     run_generation(chunks, source_docs, pages, question)
#     t2 = time.perf_counter()

#     print("\n \n This answer is based on the following context: ")
#     for i in range(len(chunks)):
#         print(
#             f"Page_Number: {pages[i]} \n Document used: {source_docs[i]} \n Chunk: {chunks[i]}"
#         )

#     print("\n--- Timing ---")
#     print(f"Retrieval: {t1 - t0:.3f}s")
#     print(f"Generation: {t2 - t1:.3f}s")


# if __name__ == "__main__":
#     app()
