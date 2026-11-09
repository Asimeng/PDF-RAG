"""Command-line interface:  rag --help"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(help="Local RAG over your PDFs, powered by Ollama.", no_args_is_help=True,
                  pretty_exceptions_show_locals=False)
console = Console()


@app.callback()
def main(
    profile: str = typer.Option(None, "--profile", "-p", help="cpu | gpu | gpu-max | auto"),
    llm: str = typer.Option(None, "--llm", help="Override the answering model, e.g. qwen3:32b"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
):
    # CLI flags become env vars so Settings picks them up everywhere.
    if profile:
        os.environ["RAG_PROFILE"] = profile
    if llm:
        os.environ["RAG_LLM_MODEL"] = llm
    from ragapp.config import get_settings
    from ragapp.logging_setup import setup_logging

    get_settings.cache_clear()
    setup_logging("DEBUG" if verbose else get_settings().log_level)


def _pipeline():
    from ragapp.pipeline import RAGPipeline

    try:
        return RAGPipeline()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def profiles():
    """List the hardware profiles and their models."""
    from ragapp.config import PROFILES

    t = Table(title="Hardware profiles")
    for col in ("name", "LLM", "embeddings", "reranker", "ctx", "needs"):
        t.add_column(col)
    for p in PROFILES.values():
        t.add_row(p.name, p.llm_model, f"{p.embed_provider}:{p.embed_model}", p.reranker_model,
                  str(p.llm_num_ctx), f">= {p.min_vram_gb:.0f} GB VRAM" if p.min_vram_gb else "CPU")
    console.print(t)


@app.command()
def status():
    """Show the active profile, models, Ollama connection and index contents."""
    pipe = _pipeline()
    st = pipe.status()
    pipe.close()
    c, o, i = st["config"], st["ollama"], st["index"]
    console.print(Panel.fit(
        f"[bold]Profile:[/bold] {c['profile']}  [dim]({c['profile_reason']})[/dim]\n"
        f"[bold]LLM:[/bold] {c['llm_model']} (ctx {c['llm_num_ctx']})\n"
        f"[bold]Embeddings:[/bold] {c['embed_provider']}:{c['embed_model']}\n"
        f"[bold]Reranker:[/bold] {c['reranker_model']}\n"
        f"[bold]Ollama:[/bold] {o['url']} - "
        + ("[green]reachable[/green]" if o["reachable"] else "[red]NOT reachable[/red]")
        + (f"\n[yellow]Missing models: {', '.join(o['missing_models'])}[/yellow]" if o["missing_models"] else "")
        + f"\n[bold]Index:[/bold] {i['collection']} - {i['documents']} documents, {i['chunks']} chunks "
        f"[dim]({i['store']})[/dim]"
        + (f"\n[dim]Other indexes: {i['other_indexes']}[/dim]" if i["other_indexes"] else ""),
        title="RAG status"))


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="PDF files and/or folders of PDFs"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-index even if unchanged"),
):
    """Add PDFs to the index (unchanged files are skipped)."""
    pipe = _pipeline()
    results = []
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(),
                  console=console, transient=True) as prog:
        task = prog.add_task("Starting...", total=1.0)
        for path in paths:
            results += pipe.ingest_path(path, force=force,
                                        progress=lambda m, f: prog.update(task, description=m, completed=f))
    pipe.close()
    t = Table(title="Ingestion results")
    for col in ("file", "status", "chunks", "seconds", "message"):
        t.add_column(col)
    colors = {"indexed": "green", "skipped": "yellow", "failed": "red"}
    for r in results:
        t.add_row(r.file_name, f"[{colors[r.status]}]{r.status}[/]", str(r.num_chunks),
                  str(r.seconds), r.message)
    console.print(t)


@app.command()
def docs():
    """List indexed documents."""
    pipe = _pipeline()
    items = pipe.list_documents()
    pipe.close()
    t = Table(title=f"Indexed documents ({len(items)})")
    for col in ("doc_id", "file", "title", "pages", "chunks", "indexed at"):
        t.add_column(col)
    for d in items:
        t.add_row(d.doc_id, d.file_name, d.title[:60], str(d.num_pages), str(d.num_chunks), d.ingested_at)
    console.print(t)


@app.command()
def delete(doc_id: str = typer.Argument(..., help="doc_id from `rag docs`")):
    """Remove a document from the index."""
    pipe = _pipeline()
    ok = pipe.delete_document(doc_id)
    pipe.close()
    console.print("[green]Deleted.[/green]" if ok else "[yellow]No such document.[/yellow]")


@app.command()
def search(query: str, k: int = typer.Option(5, "-k", help="Number of results")):
    """Show the passages retrieval finds for a query (no LLM involved)."""
    pipe = _pipeline()
    hits = pipe.search(query, top_k=k)
    pipe.close()
    for n, h in enumerate(hits, 1):
        c = h.chunk
        score = f"rerank {h.rerank_score:.3f}" if h.rerank_score is not None else f"rrf {h.score:.3f}"
        console.print(Panel(c.text[:900], title=f"[{n}] {c.file_name} {c.pages_label} | {c.section[:50]}",
                            subtitle=score))


def _print_answer_stream(pipe, question: str, history: list | None = None) -> dict | None:
    from ragapp.generate.llm import LLMError

    answer = None
    try:
        for ev in pipe.ask_stream(question, history=history):
            if ev["type"] == "sources" and ev["standalone_question"] != question:
                console.print(f"[dim]Searching for: {ev['standalone_question']}[/dim]")
            elif ev["type"] == "token":
                console.print(ev["text"], end="", markup=False, highlight=False)
            elif ev["type"] == "done":
                answer = ev["answer"]
    except (LLMError, RuntimeError) as exc:
        console.print(f"\n[red]{exc}[/red]")
        return None
    console.print("\n")
    if answer and answer["sources"]:
        t = Table(title="Sources", show_lines=False)
        for col in ("#", "file", "pages", "section", "score"):
            t.add_column(col)
        for s in answer["sources"]:
            pages = str(s["page_start"]) if s["page_start"] == s["page_end"] else f"{s['page_start']}-{s['page_end']}"
            mark = "[bold green]*[/]" if s["cited"] else " "
            t.add_row(f"{mark}{s['ref']}", s["file_name"], pages, s["section"][:40], f"{s['score']:.3f}")
        console.print(t)
        tm = answer["timings_ms"]
        console.print(f"[dim]model {answer['model']} | retrieve {tm.get('retrieve', 0):.0f} ms | "
                      f"first token {tm.get('first_token', 0):.0f} ms | total generate "
                      f"{tm.get('generate', 0) / 1000:.1f} s   (* = cited)[/dim]")
    return answer


@app.command()
def ask(question: str):
    """Ask one question and get a cited answer."""
    pipe = _pipeline()
    try:
        _print_answer_stream(pipe, question)
    finally:
        pipe.close()


@app.command()
def chat():
    """Interactive chat with follow-up questions (type 'exit' to quit)."""
    pipe = _pipeline()
    history: list[dict] = []
    console.print(Markdown("**Chat with your documents.** Type `exit` to quit, `reset` to clear history."))
    try:
        while True:
            q = console.input("[bold cyan]You:[/] ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q.lower() == "reset":
                history.clear()
                continue
            if not q:
                continue
            console.print("[bold magenta]Assistant:[/] ", end="")
            ans = _print_answer_stream(pipe, q, history)
            if ans:
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans["answer"]}]
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        pipe.close()


@app.command("eval")
def evaluate(
    questions: Path = typer.Option(Path("eval/questions.jsonl"), exists=True, help="JSONL test set"),
    out: Path = typer.Option(Path("eval/results"), help="Where to write the report"),
    answers: bool = typer.Option(False, "--answers", help="Also generate & score answers (needs Ollama)"),
    judge: str = typer.Option(None, "--judge", help="Ollama model used as faithfulness judge"),
):
    """Measure retrieval quality (and optionally answer quality) on a test set."""
    from ragapp.evaluate import run

    with console.status("Evaluating... (answer generation can take a while)"):
        report = run(questions, out, generate=answers or bool(judge), judge_model=judge)
    console.print(Markdown(report.read_text(encoding="utf-8").split("<details>")[0]))
    console.print(f"[dim]Full report: {report}[/dim]")


@app.command()
def serve(host: str = typer.Option(None), port: int = typer.Option(None)):
    """Start the REST API server (docs at http://host:port/docs)."""
    import uvicorn

    from ragapp.config import get_settings

    s = get_settings()
    uvicorn.run("ragapp.api:app", host=host or s.api_host, port=port or s.api_port, log_level="info")


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket() as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _free_port(host: str) -> int:
    import socket

    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@app.command()
def gui(
    host: str = typer.Option(None, help="Address for the GUI (0.0.0.0 = reachable from other computers)"),
    port: int = typer.Option(None, help="GUI port (default 8501)"),
    start_api: bool = typer.Option(True, "--start-api/--no-start-api",
                                   help="Start the API in the background if it isn't running"),
):
    """Open the web interface (starts the API too, if needed)."""
    import subprocess
    import sys
    import time

    import httpx

    from ragapp.config import get_settings

    s = get_settings()
    url = s.resolved_api_url

    def healthy() -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    api_proc = None
    if not healthy():
        if not start_api or s.api_url:
            console.print(f"[red]The API at {url} is not reachable.[/red] Start it with: rag serve")
            raise typer.Exit(1)
        api_host, api_port = s.api_host, s.api_port
        if not _port_free(api_host, api_port):  # e.g. another program already uses 8000
            api_port = _free_port(api_host)
            console.print(f"[yellow]Port {s.api_port} is in use by another program; "
                          f"using {api_port} instead.[/yellow]")
        url = f"http://{'127.0.0.1' if api_host in {'0.0.0.0', '::'} else api_host}:{api_port}"
        os.environ["RAG_API_URL"] = url  # tells the GUI process where the API is
        console.print(f"Starting the API at {url} ...")
        api_proc = subprocess.Popen([sys.executable, "-m", "ragapp.cli", "serve",
                                     "--host", api_host, "--port", str(api_port)])
        for _ in range(120):
            if healthy():
                break
            if api_proc.poll() is not None:
                console.print("[red]The API failed to start - see the messages above.[/red]")
                raise typer.Exit(1)
            time.sleep(1)
        else:
            api_proc.terminate()
            console.print("[red]The API did not start within 2 minutes.[/red]")
            raise typer.Exit(1)

    app_path = Path(__file__).parent / "gui" / "app.py"
    gui_host, gui_port = host or s.gui_host, port or s.gui_port
    if not _port_free(gui_host, gui_port):  # e.g. a forwarded port from a remote copy of the app
        busy, gui_port = gui_port, _free_port(gui_host)
        console.print(f"[yellow]Port {busy} is in use by another program; "
                      f"using {gui_port} instead.[/yellow]")
    console.print(f"[green]Opening the GUI at http://{'localhost' if gui_host in {'127.0.0.1', '0.0.0.0'} else gui_host}:{gui_port}[/green]  (Ctrl+C to stop)")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.address", gui_host, "--server.port", str(gui_port),
           "--server.maxUploadSize", str(s.max_upload_mb),
           "--browser.gatherUsageStats", "false",
           "--client.toolbarMode", "minimal",
           "--theme.primaryColor", "#2563eb"]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass
    finally:
        if api_proc is not None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()


if __name__ == "__main__":
    app()
