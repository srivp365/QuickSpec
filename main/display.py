from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


def render_answer_panel(question: str, answer: str, sources: list[dict]):
    """sources: list of dicts with keys 'source_doc', 'page_number', 'text'"""
    console.print(Panel(answer, title=f"[bold cyan]{question}[/bold cyan]", border_style="cyan"))

    if sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("Doc", style="dim")
        table.add_column("Page", justify="right")
        table.add_column("Preview")

        for s in sources:
            preview = s["text"][:80].replace("\n", " ") + "..."
            table.add_row(s["source_doc"], str(s["page_number"]), preview)

        console.print(table)


def render_eval_table(results_by_config: dict[str, dict]):
    """results_by_config: {"vector": {"precision@5": ..., "mrr": ..., "gen_accuracy": ...}, ...}"""
    table = Table(title="Retrieval Configuration Comparison")
    table.add_column("Config", style="bold")
    table.add_column("Precision@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Gen Accuracy", justify="right")

    for config_name, metrics in results_by_config.items():
        precision_key = next((k for k in metrics if k.startswith("precision")), None)
        table.add_row(
            config_name,
            f"{metrics.get(precision_key, 0):.3f}" if precision_key else "-",
            f"{metrics.get('mrr', 0):.3f}",
            f"{metrics.get('gen_accuracy', 0):.3f}",
        )

    console.print(table)


def render_stats_table(stats: dict):
    """stats: {'total_chunks': int, 'by_type': {'prose': n, 'table': n}, 'source_docs': [str]}"""
    table = Table(title="Corpus Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total chunks", str(stats["total_chunks"]))
    for chunk_type, count in stats.get("by_type", {}).items():
        table.add_row(f"  {chunk_type}", str(count))
    table.add_row("Source docs", str(len(stats.get("source_docs", []))))

    console.print(table)

    if stats.get("source_docs"):
        doc_table = Table(title="Documents")
        doc_table.add_column("Path")
        for doc in stats["source_docs"]:
            doc_table.add_row(doc)
        console.print(doc_table)


def make_progress() -> Progress:
    """Rich progress bar matching the style used elsewhere in the CLI.
    Use as: with make_progress() as progress: task = progress.add_task(...); progress.update(task, advance=1)"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def print_error(message: str):
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str):
    console.print(f"[bold green]✓[/bold green] {message}")
