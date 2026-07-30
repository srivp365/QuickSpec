import typer
import time
from main.retrieval.retrieval import run_retrieval
from main.generation.generation import run_generation
from main.indexing.indexing import get_files

app = typer.Typer()


@app.command()
def load_documents(file_path):
    start = time.perf_counter()

    t0 = time.perf_counter()
    get_files(file_path)
    t1 = time.perf_counter()
    print()

    print("\n--- Timing ---")
    print(f"Indexing: {t1 - t0:.3f}s")



@app.command()
def answer_question(question):
    start = time.perf_counter()

    t0 = time.perf_counter()
    chunks = run_retrieval(question)
    t1 = time.perf_counter()
    run_generation(chunks, question)
    t2 = time.perf_counter()

    print()

    print("\n--- Timing ---")
    print(f"Retrieval: {t1 - t0:.3f}s")
    print(f"Generation: {t2 - t1:.3f}s")

if __name__ == "__main__":
    app()
