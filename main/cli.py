import sys
import time

import typer

from main.generation.generation import run_generation
from main.indexing.indexing import get_files
from main.retrieval.retrieval import run_retrieval

app = typer.Typer()


@app.command()
def load_documents(file_path):
    t0 = time.perf_counter()
    get_files(file_path)
    t1 = time.perf_counter()

    print("\n--- Timing ---")
    print(f"Indexing: {t1 - t0:.3f}s")
    sys.exit(0)


@app.command()
def ask(question):
    t0 = time.perf_counter()
    chunks, source_docs, pages = run_retrieval(question)
    t1 = time.perf_counter()
    run_generation(chunks, question)
    t2 = time.perf_counter()

    print("\n \n This answer is based on the following context: ")
    for i in range(len(chunks)):
        print(
            f"Page_Number: {pages[i]} \n Document used: {source_docs[i]} \n Chunk: {chunks[i]}"
        )

    print("\n--- Timing ---")
    print(f"Retrieval: {t1 - t0:.3f}s")
    print(f"Generation: {t2 - t1:.3f}s")


if __name__ == "__main__":
    app()
