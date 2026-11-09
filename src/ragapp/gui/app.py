"""Web interface for non-technical users (Streamlit).

Start it with:  rag gui    (this also starts the API if it isn't running)

Three pages:
  * Ask        - chat with your documents; every answer shows its sources, and
                 each source opens the real PDF page with the passage highlighted.
  * Documents  - add PDFs (drag and drop), see what's in the library, remove files.
  * System     - which models are in use, whether Ollama is reachable, and a
                 model picker.

The GUI only talks to the REST API (see client.py), never to the models directly.
Streamlit re-runs this script from the top on every click; anything that must
survive a re-run lives in ``st.session_state``.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

import streamlit as st

from ragapp.config import get_settings
from ragapp.generate.prompts import NOT_FOUND
from ragapp.gui.client import APIError, RAGClient

st.set_page_config(page_title="Ask your PDFs", page_icon=":material/description:", layout="wide")


# ======================================================================= helpers
@st.cache_resource
def get_client() -> RAGClient:
    s = get_settings()
    return RAGClient(s.resolved_api_url, s.api_key)


client = get_client()


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def page_image(doc_id: str, page: int, highlight: str) -> bytes:
    return client.page_image(doc_id, page, highlight)


@st.cache_data(ttl=600, show_spinner=False, max_entries=20)
def document_file(doc_id: str) -> bytes:
    return client.document_file(doc_id)


@st.cache_data(ttl=5, show_spinner=False)
def documents() -> list[dict]:
    return client.documents()


def refresh_documents() -> None:
    documents.clear()


_CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")


def pretty_answer(text: str) -> str:
    """Make the answer render safely and turn [1] citations into small badges."""
    text = text.replace("$", r"\$")  # Streamlit treats $...$ as maths
    return _CITE.sub(lambda m: f":blue-badge[{m.group(1)}]", text)


def is_refusal(text: str) -> bool:
    return text.strip().startswith(NOT_FOUND[:40])


def source_label(s: dict) -> str:
    pages = f"p. {s['page_start']}" if s["page_start"] == s["page_end"] \
        else f"pp. {s['page_start']}-{s['page_end']}"
    title = s["title"] if len(s["title"]) <= 40 else s["title"][:38].rstrip() + "…"
    return f"[{s['ref']}] {pages} · {title}"


def show_error(exc: Exception) -> None:
    st.error(str(exc), icon=":material/error:")


# ======================================================================= Ask page
def render_sources(msg: dict, idx: int) -> None:
    """Source picker + the cited PDF page with the passage highlighted."""
    sources = msg.get("sources") or []
    if not sources or is_refusal(msg["content"]):
        return
    cited = [s for s in sources if s.get("cited")]
    shown = cited or sources
    by_ref = {s["ref"]: s for s in shown}
    title = f"Sources: {len(cited)} cited" if cited else f"Sources: {len(sources)} passages used"
    with st.expander(title, icon=":material/menu_book:", expanded=idx == len(st.session_state.messages) - 1):
        ref = st.pills("Source", list(by_ref), format_func=lambda r: source_label(by_ref[r]),
                       default=shown[0]["ref"], key=f"src_{idx}", label_visibility="collapsed")
        s = by_ref.get(ref) or shown[0]
        left, right = st.columns([3, 2], gap="large")
        with right:
            st.markdown(f"**{s['title']}**")
            st.caption(f"{s['file_name']}" + (f" | Section: {s['section']}" if s["section"] else ""))
            pages = list(range(s["page_start"], s["page_end"] + 1))
            page = pages[0]
            if len(pages) > 1:
                page = st.segmented_control("Page", pages, default=pages[0], key=f"pg_{idx}_{s['ref']}",
                                            format_func=lambda p: f"Page {p}") or pages[0]
            with st.container(height=360, border=True):
                st.markdown(s["text"].replace("$", r"\$"))
            if s.get("doc_id"):
                try:
                    st.download_button("Download PDF", document_file(s["doc_id"]), file_name=s["file_name"],
                                       mime="application/pdf", icon=":material/download:",
                                       key=f"dl_{idx}_{s['ref']}", on_click="ignore")
                except APIError:
                    pass
        with left:
            if not s.get("doc_id"):
                st.info("Page preview is unavailable for this source.")
            else:
                try:
                    with st.spinner("Loading page..."):
                        st.image(page_image(s["doc_id"], page, s["text"]), width="stretch",
                                 caption=f"{s['file_name']}, page {page} (highlighted: the passage used)")
                except APIError as exc:
                    st.info(f"Page preview unavailable: {exc}")


def render_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:"):
        if msg["role"] == "user":
            st.markdown(msg["content"].replace("$", r"\$"))
            return
        if msg.get("error"):
            st.error(msg["content"], icon=":material/error:")
            return
        st.markdown(pretty_answer(msg["content"]))
        meta = msg.get("meta") or {}
        if meta:
            bits = [f"Model: {meta.get('model')}"]
            if meta.get("seconds") is not None:
                bits.append(f"{meta['seconds']:.1f} s")
            if meta.get("standalone") and meta["standalone"] != meta.get("question"):
                bits.append(f"Searched for: “{meta['standalone']}”")
            st.caption(" | ".join(bits))
        render_sources(msg, idx)


def history_for_api() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages if not m.get("error")][-6:]


def run_question(question: str, doc_ids: list[str] | None) -> None:
    history = history_for_api()
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        status = st.status("Searching your documents...", expanded=False)
        final: dict = {}
        started = time.perf_counter()

        def tokens():
            for ev in client.ask_stream(question, history, doc_ids):
                if ev["type"] == "sources":
                    n = len(ev["sources"])
                    status.update(label=f"Found {n} relevant passage{'s' * (n != 1)}. Writing the answer...")
                elif ev["type"] == "token":
                    yield ev["text"]
                elif ev["type"] == "done":
                    final.update(ev["answer"])
                elif ev["type"] == "error":
                    raise APIError(ev["detail"])

        try:
            st.write_stream(tokens())
            status.update(label="Done", state="complete")
            st.session_state.messages.append({
                "role": "assistant", "content": final.get("answer", ""), "sources": final.get("sources", []),
                "meta": {"model": final.get("model"), "question": question,
                         "standalone": final.get("standalone_question"),
                         "seconds": time.perf_counter() - started},
            })
        except APIError as exc:
            status.update(label="Something went wrong", state="error")
            st.session_state.messages.append({"role": "assistant", "content": str(exc), "error": True})
    st.rerun()  # redraw with the formatted answer and its sources


def transcript_markdown() -> str:
    lines = [f"# Conversation ({datetime.now():%Y-%m-%d %H:%M})", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
        for s in [s for s in m.get("sources") or [] if s.get("cited")]:
            lines.append(f"- {source_label(s)} ({s['file_name']})")
        lines.append("")
    return "\n".join(lines)


def _use_example() -> None:
    st.session_state.pending = st.session_state.example
    st.session_state.example = None  # so the example isn't asked again on every re-run


def ask_page() -> None:
    st.session_state.setdefault("messages", [])
    try:
        docs = documents()
    except APIError as exc:
        show_error(exc)
        return

    with st.sidebar:
        st.subheader("Search in")
        titles = {d["doc_id"]: d["title"] for d in docs}
        selected = st.multiselect("Documents", list(titles), format_func=lambda d: titles[d],
                                  placeholder="All documents", label_visibility="collapsed",
                                  help="Leave empty to search every document.")
        c1, c2 = st.columns(2)
        if c1.button("New chat", icon=":material/add_comment:", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        c2.download_button("Save chat", transcript_markdown(), file_name="conversation.md",
                           mime="text/markdown", icon=":material/save:", width="stretch",
                           disabled=not st.session_state.messages, on_click="ignore")

    st.title("Ask your documents")
    if not docs:
        st.info("Your library is empty. Add some PDFs first, then come back to ask questions.",
                icon=":material/info:")
        st.page_link(PAGES["documents"], label="Go to Documents", icon=":material/upload_file:")
        return

    if not st.session_state.messages:
        st.markdown(
            f"Ask a question in plain language about your **{len(docs)} document"
            f"{'s' * (len(docs) != 1)}**. Every answer lists its sources, and you can open "
            "each source to see the original page with the passage highlighted."
        )
        examples = [f"What are the main findings of “{d['title'][:70]}”?" for d in docs[:2]]
        examples.append("Summarise the methods used in these documents.")
        st.pills("Try an example", examples, key="example", on_change=_use_example)

    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)

    question = st.chat_input("Ask a question about your documents...")
    question = question or st.session_state.pop("pending", None)
    if question:
        run_question(question.strip(), selected or None)


# ======================================================================= Documents page
@st.dialog("Remove documents?")
def confirm_delete(docs: list[dict]) -> None:
    st.write("These documents will no longer be searched:")
    for d in docs:
        st.markdown(f"- {d['title']} (`{d['file_name']}`)")
    delete_file = st.checkbox("Also delete the PDF files from the library folder")
    if st.button("Remove", type="primary", icon=":material/delete:"):
        try:
            for d in docs:
                client.delete_document(d["doc_id"], delete_file=delete_file)
        except APIError as exc:
            show_error(exc)
            return
        refresh_documents()
        st.session_state.toast = f"Removed {len(docs)} document{'s' * (len(docs) != 1)}."
        st.rerun()


def upload_section() -> None:
    st.subheader("Add PDFs")
    st.session_state.setdefault("uploader_key", 0)
    files = st.file_uploader("Drop PDF files here", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader_{st.session_state.uploader_key}",
                             help="Text-based PDFs work best. Scanned images without text can't be read yet.")
    if not files:
        return
    if not st.button(f"Add {len(files)} file{'s' * (len(files) != 1)} to the library",
                     type="primary", icon=":material/upload:"):
        return
    try:
        job = client.upload([(f.name, f.getvalue()) for f in files])
    except APIError as exc:
        show_error(exc)
        return
    bar = st.progress(0.0, text="Preparing...")
    while job["status"] in {"queued", "running"}:
        time.sleep(1.0)
        try:
            job = client.job(job["job_id"])
        except APIError as exc:
            show_error(exc)
            return
        bar.progress(min(float(job["progress"]), 1.0), text=job["message"])
    bar.empty()
    if job["status"] == "failed":
        st.error(f"Adding documents failed: {job['message']}", icon=":material/error:")
        return
    for r in job["results"]:
        if r["status"] == "indexed":
            note = f" Note: {r['message']}." if r.get("message") else ""
            st.success(f"Added **{r['file_name']}** ({r.get('num_chunks')} passages). Ready to search.{note}",
                       icon=":material/check_circle:")
        elif r["status"] == "skipped":
            st.info(f"**{r['file_name']}** is already in the library (unchanged).", icon=":material/info:")
        else:
            st.error(f"Couldn't add **{r['file_name']}**: {r.get('message') or 'unknown error'}",
                     icon=":material/error:")
    refresh_documents()
    st.session_state.uploader_key += 1  # clear the drop zone on the next interaction


def documents_page() -> None:
    st.title("Documents")
    if msg := st.session_state.pop("toast", None):
        st.toast(msg, icon=":material/check:")
    upload_section()
    st.divider()

    try:
        docs = documents()
    except APIError as exc:
        show_error(exc)
        return
    st.subheader(f"Library ({len(docs)})")
    if not docs:
        st.caption("No documents yet.")
        return
    rows = [{"Title": d["title"], "File": d["file_name"], "Pages": d["num_pages"],
             "Passages": d["num_chunks"], "Added": d["ingested_at"][:16].replace("T", " ")} for d in docs]
    event = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="multi-row", key="doc_table",
                         column_config={"Title": st.column_config.TextColumn(width="large"),
                                        "Passages": st.column_config.NumberColumn(
                                            help="Number of searchable text pieces")})
    chosen = [docs[i] for i in event.selection.rows]
    if not chosen:
        st.caption("Select documents in the table to download or remove them.")
        return
    c1, c2, _ = st.columns([1, 1, 3])
    if len(chosen) == 1:
        try:
            c1.download_button("Download PDF", document_file(chosen[0]["doc_id"]),
                               file_name=chosen[0]["file_name"], mime="application/pdf",
                               icon=":material/download:", on_click="ignore", width="stretch")
        except APIError as exc:
            c1.caption(str(exc))
    if c2.button(f"Remove {len(chosen)}", icon=":material/delete:", width="stretch"):
        confirm_delete(chosen)


# ======================================================================= System page
def system_page() -> None:
    st.title("System")
    try:
        status = client.status()
    except APIError as exc:
        show_error(exc)
        return
    cfg, oll, idx = status["config"], status["ollama"], status["index"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Hardware profile", cfg["profile"])
    c2.metric("Documents", idx["documents"])
    c3.metric("Searchable passages", idx["chunks"])
    st.caption(f"Why this profile: {cfg['profile_reason']}")

    if not oll["reachable"]:
        st.error(f"Ollama is not reachable at {oll['url']}. Searching works, but answers can't be "
                 "written until Ollama is running.", icon=":material/cloud_off:")
    elif oll["missing_models"]:
        st.warning("These models are not installed on Ollama: " + ", ".join(oll["missing_models"])
                   + f". Install with: `ollama pull {oll['missing_models'][0]}`", icon=":material/warning:")
    else:
        st.success(f"Ollama is running at {oll['url']}.", icon=":material/check_circle:")

    st.subheader("Answering model")
    st.write("The language model that writes the answers. Larger models give better answers but are slower.")
    installed: list[str] = []
    if oll["reachable"]:
        try:  # /models hides cloud models unless they are allowed
            installed = [m for m in client.models()["models"] if "embed" not in m]
        except APIError:
            pass
    if installed:
        current = cfg["llm_model"]
        options = installed if current in installed else [current, *installed]
        choice = st.selectbox("Model", options, index=options.index(current))
        if choice != current and st.button("Use this model", type="primary"):
            try:
                client.set_llm(choice)
                st.toast(f"Now answering with {choice}", icon=":material/check:")
                st.rerun()
            except APIError as exc:
                show_error(exc)
        st.caption("This applies to everyone using this server until it restarts.")
    else:
        st.write(f"Current model: `{cfg['llm_model']}`")

    st.subheader("Configuration")
    st.dataframe(
        [{"Setting": k, "Value": str(v)} for k, v in [
            ("Language model", cfg["llm_model"]), ("Context window (tokens)", cfg["llm_num_ctx"]),
            ("Embedding model", f"{cfg['embed_provider']}: {cfg['embed_model']}"),
            ("Reranker", cfg["reranker_model"]), ("Passages searched", cfg["top_k_retrieve"]),
            ("Passages given to the model", cfg["top_k_context"]), ("Index", idx["collection"]),
            ("Storage", idx["store"]), ("Service URL", client.base_url)]],
        hide_index=True, width="stretch")
    if idx["other_indexes"]:
        st.info("Documents also exist in other indexes built with different embedding models: "
                + ", ".join(f"{k} ({v} docs)" for k, v in idx["other_indexes"].items())
                + ". They are not searched with the current profile.", icon=":material/info:")


# ======================================================================= layout
PAGES = {
    "ask": st.Page(ask_page, title="Ask", icon=":material/chat:", default=True),
    "documents": st.Page(documents_page, title="Documents", icon=":material/folder_open:",
                         url_path="documents"),
    "system": st.Page(system_page, title="System", icon=":material/settings:", url_path="system"),
}

with st.sidebar:
    st.markdown("### :material/description: Ask your PDFs")
    if not client.health():
        st.error(f"Can't reach the service at {client.base_url}.", icon=":material/cloud_off:")
        st.caption("Start everything with `rag gui`, or set RAG_API_URL to the server address.")
        st.stop()

st.navigation(list(PAGES.values())).run()
