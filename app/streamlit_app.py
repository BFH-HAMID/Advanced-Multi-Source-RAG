"""Streamlit UI for the Advanced Multi-Source RAG knowledge base.

Features
--------
* upload CSV / PDF / TXT / MD files and add URLs,
* one-click demo data + index building with live progress,
* chat interface with streaming answers,
* bullet citations ``[file_name, page]`` and an expandable sources panel,
* knowledge-base browser and RAGAS evaluation runner.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import configure_logging, get_settings  # noqa: E402
from src.generation.formatter import bullet_citations, group_contexts_by_source  # noqa: E402
from src.indexing.build_all import IndexBuilder  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.pipeline import RAGPipeline, build_pipeline  # noqa: E402
from src.storage.document_db import DocumentDB  # noqa: E402

SUPPORTED_UPLOADS = ["pdf", "csv", "tsv", "txt", "md"]
CITATION_HINT = "Answers cite their sources as `[file_name, page]`."


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_settings_cached() -> Any:
    """Load settings once per process."""
    configure_logging("INFO")
    return get_settings()


@st.cache_resource(show_spinner=False)
def get_db(version: int) -> DocumentDB:
    """Document DB handle (``version`` invalidates the cache after ingestion)."""
    return DocumentDB(settings=get_settings_cached())


@st.cache_resource(show_spinner=False)
def get_ingestion_pipeline(version: int) -> IngestionPipeline:
    """Ingestion pipeline (re-created when the KB version changes)."""
    settings = get_settings_cached()
    return IngestionPipeline(settings=settings, db=get_db(version))


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(version: int) -> RAGPipeline:
    """Query pipeline; indexes are loaded lazily on the first question."""
    return build_pipeline(get_settings_cached(), db=get_db(version))


@st.cache_resource(show_spinner=False)
def get_index_builder(version: int) -> IndexBuilder:
    """Index builder sharing the document DB."""
    return IndexBuilder(settings=get_settings_cached(), db=get_db(version))


def bump_version() -> None:
    """Invalidate cached pipelines after a mutation."""
    st.session_state["kb_version"] = int(st.session_state.get("kb_version", 0)) + 1


def current_version() -> int:
    return int(st.session_state.setdefault("kb_version", 0))


# --------------------------------------------------------------------------- #
# Sidebar: sources, ingestion, status
# --------------------------------------------------------------------------- #
def render_status(settings: Any, db: DocumentDB) -> None:
    """Show model/index status and warnings."""
    stats = db.get_stats()
    indexes = {
        "Vector (FAISS)": settings.vector_index_dir.exists(),
        "Sentence-window": settings.window_index_dir.exists(),
        "Graph (NetworkX)": settings.graph_path.exists(),
    }
    st.markdown("#### System status")
    provider = settings.llm_provider
    model = settings.resolved_llm_model
    st.markdown(
        f"**LLM:** `{provider}` · `{model}`  \n"
        f"**Embeddings:** `{settings.embedding_backend}` · `{settings.embedding_model}`  \n"
        f"**Reranker:** `{settings.reranker_backend}` · `{settings.reranker_model}`"
    )
    if provider in {"openai", "anthropic"} and not (
        settings.openai_api_key or settings.anthropic_api_key
    ):
        st.warning(
            "No API key configured for the selected provider — answers use the "
            "grounded extractive fallback. Set OPENAI_API_KEY / ANTHROPIC_API_KEY in `.env`.",
            icon="⚠️",
        )
    for label, exists in indexes.items():
        st.caption(f"{'✅' if exists else '⬜'} {label}")
    st.caption(
        f"📄 {stats['documents']} documents · 🧩 {stats['chunks']} chunks · "
        f"{stats['tokens']:,} tokens"
    )


def render_ingestion(settings: Any, db: DocumentDB, version: int) -> None:
    """File upload, URL input and index building controls."""
    pipeline = get_ingestion_pipeline(version)
    st.markdown("#### 1 · Add documents")
    uploads = st.file_uploader(
        "Upload PDF / CSV / TXT / MD files",
        type=SUPPORTED_UPLOADS,
        accept_multiple_files=True,
        help="Files are stored in data/raw, cleaned, chunked (600 tokens / 80 overlap) and indexed.",
    )
    if uploads and st.button("📥 Ingest uploaded files", width="stretch"):
        with st.status("Ingesting files…", expanded=True) as status:
            paths = []
            for upload in uploads:
                try:
                    paths.append(pipeline.save_upload(upload.name, upload.getbuffer().tobytes()))
                    st.write(f"• saved {upload.name}")
                except Exception as exc:  # pragma: no cover - UI guard
                    st.error(f"{upload.name}: {exc}")
            if paths:
                report = pipeline.ingest_files(paths)
                st.session_state["last_report"] = report.to_dict()
                _show_report(report)
                bump_version()
                status.update(label="Ingestion finished", state="complete")
            else:
                status.update(label="Nothing to ingest", state="error")

    with st.form("url_form", clear_on_submit=False):
        url_text = st.text_area(
            "Add web pages (one URL per line)",
            placeholder="https://example.com/hr-policies",
            height=90,
        )
        submitted = st.form_submit_button("🌐 Ingest URLs", width="stretch")
    if submitted and url_text.strip():
        with st.status("Fetching pages…", expanded=True) as status:
            report = pipeline.ingest_urls(url_text.split())
            st.session_state["last_report"] = report.to_dict()
            _show_report(report)
            bump_version()
            status.update(label="URL ingestion finished", state="complete")

    if st.button("🎬 Load demo knowledge base", width="stretch", help="Ingests data/samples"):
        samples = sorted((settings.data_dir / "samples").iterdir())
        with st.status("Loading demo data…", expanded=True) as status:
            report = pipeline.ingest_files(samples)
            st.session_state["last_report"] = report.to_dict()
            _show_report(report)
            bump_version()
            status.update(label="Demo data loaded", state="complete")

    st.divider()
    st.markdown("#### 2 · Build indexes")
    builder = get_index_builder(version)
    existing = builder.indexes_exist()
    st.caption(
        " · ".join(f"{name}: {'✅' if ok else '⬜'}" for name, ok in existing.items())
        + (" · ⚠️ stale" if db.count_chunks() and builder.is_stale() else "")
    )
    skip_graph = st.checkbox("Skip graph index (faster)", value=False)
    if st.button("🏗️ Build / rebuild indexes", type="primary", width="stretch"):
        progress_bar = st.progress(0.0, text="Starting…")

        def on_progress(stage: str, done: int, total: int) -> None:
            fraction = min(1.0, done / max(1, total)) if total else 0.0
            progress_bar.progress(fraction, text=f"{stage}: {done}/{total}")

        started = time.perf_counter()
        try:
            report = builder.build_all(skip_graph=skip_graph, progress=on_progress)
            st.session_state["last_build"] = report.to_dict()
            st.success(
                f"Indexed {report.vector_nodes} chunks · {report.window_nodes} window nodes · "
                f"{report.graph_entities} entities / {report.graph_relations} relations "
                f"in {time.perf_counter() - started:.1f}s"
            )
            if report.errors:
                with st.expander("Warnings"):
                    for error in report.errors:
                        st.warning(error)
            bump_version()
        except Exception as exc:
            st.error(f"Index build failed: {exc}")

    with st.expander("🧹 Maintenance"):
        if st.button("Re-chunk documents from SQLite"):
            from src.processing.processor import DocumentProcessor

            try:
                chunks, stats = DocumentProcessor(settings).reprocess_from_db(db)
                st.success(f"Re-chunked into {len(chunks)} chunks ({stats.tokens} tokens)")
                bump_version()
            except Exception as exc:
                st.error(str(exc))
        confirm = st.checkbox("I understand this deletes all documents and indexes")
        if st.button("Delete knowledge base", disabled=not confirm):
            db.clear_all()
            for path in (settings.vector_index_dir, settings.window_index_dir, settings.graph_dir):
                for file in path.glob("*"):
                    file.unlink()
            st.success("Knowledge base cleared")
            bump_version()


def _show_report(report: Any) -> None:
    """Render an ingestion report inside the status container."""
    st.write(
        f"**{len(report.sources)} source(s)** · {report.documents} documents · "
        f"{report.chunks} chunks · {report.tokens:,} tokens"
    )
    for detail in report.details:
        icon = "✅" if detail.status == "ok" else "❌"
        st.write(f"{icon} `{detail.source}` — {detail.documents} docs, {detail.chunks} chunks")
    for error in report.errors:
        st.error(error)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def render_sources(answer: Any) -> None:
    """Bullet citations + expandable source panel."""
    if answer.citations:
        st.markdown("**Citations**")
        for bullet in bullet_citations(answer.citations):
            st.markdown(bullet)

    contexts = answer.contexts
    if not contexts:
        return
    grouped = group_contexts_by_source(contexts)
    with st.expander(f"📚 Sources — {len(contexts)} passages from {len(grouped)} file(s)"):
        for source, items in grouped.items():
            st.markdown(f"**{source}**")
            for context in items:
                readers = ", ".join(context.retrievers) if context.retrievers else context.retriever
                score = context.rerank_score if context.rerank_score is not None else context.score
                st.caption(
                    f"`{context.citation_key}` · via {readers} · "
                    f"score {float(score or 0):.3f} · {len(context.text)} chars"
                )
                st.markdown(
                    f"> {context.text.strip()[:700].replace(chr(10), ' ')}"
                    + ("…" if len(context.text) > 700 else "")
                )
        with st.expander("Retrieval diagnostics"):
            st.json(
                {
                    "timings_ms": answer.timings_ms,
                    "per_retriever": answer.retriever_stats,
                    "provider": answer.provider,
                    "model": answer.model,
                    "errors": answer.errors,
                }
            )


def run_query(pipeline: RAGPipeline, query: str) -> dict[str, Any] | None:
    """Prepare, stream and finalize one query (rendered in the chat)."""
    with st.chat_message("assistant"):
        prepared = pipeline.prepare(query)
        if not prepared.has_context:
            answer = pipeline.answer(query)
            st.markdown(answer.answer)
            if answer.errors:
                with st.expander("Diagnostics"):
                    st.json({"errors": answer.errors, "gate_threshold": prepared.gate_threshold})
            return {"role": "assistant", "content": answer.answer, "answer": answer.to_dict()}

        placeholder = st.empty()
        collected: list[str] = []
        started = time.perf_counter()
        for chunk in pipeline.stream(prepared):
            collected.append(chunk)
            placeholder.markdown("".join(collected) + "▌")
        text = "".join(collected).strip()
        placeholder.markdown(text or "_No answer generated._")

        generation_ms = (time.perf_counter() - started) * 1000
        answer = pipeline.finalize(prepared, text, generation_ms=generation_ms)
        render_sources(answer)
        st.caption(
            f"⏱️ {answer.timings_ms.get('total_ms', 0):.0f} ms · "
            f"retrieval {answer.timings_ms.get('retrieval_ms', 0):.0f} ms · "
            f"rerank {answer.timings_ms.get('rerank_ms', 0):.0f} ms · "
            f"{answer.provider}/{answer.model}"
        )
        return {"role": "assistant", "content": answer.answer, "answer": answer.to_dict()}


def render_chat(pipeline: RAGPipeline, db: DocumentDB) -> None:
    """Chat interface with history."""
    history: list[dict[str, Any]] = st.session_state.setdefault("messages", [])
    st.markdown("#### 💬 Ask your knowledge base")
    st.caption(
        "Answers are generated **only** from your documents. "
        + CITATION_HINT
        + " Ask in English or Bangla (বাংলা)।"
    )

    if not db.count_chunks():
        st.info("No documents yet. Add files or load the demo knowledge base in the sidebar.", icon="📂")

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("answer"):
                answer_dict = message["answer"]
                citations = answer_dict.get("citations") or []
                if citations:
                    st.markdown("**Citations**")
                    for citation in citations:
                        st.markdown(f"- **{citation.get('key')}** — {citation.get('snippet', '')[:200]}")
                contexts = answer_dict.get("contexts") or []
                if contexts:
                    with st.expander(f"📚 Sources — {len(contexts)} passages"):
                        for context in contexts:
                            readers = ", ".join(context.get("retrievers") or [])
                            st.caption(
                                f"`{context.get('citation_key')}` · via {readers} · "
                                f"score {float(context.get('rerank_score') or 0):.3f}"
                            )
                            st.markdown(f"> {str(context.get('text', ''))[:600]}")

    query = st.chat_input("Ask a question about your documents…")
    if query:
        history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        try:
            payload = run_query(pipeline, query)
        except Exception as exc:  # pragma: no cover - UI guard
            payload = {"role": "assistant", "content": f"⚠️ {exc}"}
            with st.chat_message("assistant"):
                st.error(str(exc))
        if payload:
            history.append(payload)
            st.session_state["messages"] = history

    if history and st.button("🗑️ Clear chat"):
        st.session_state["messages"] = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Knowledge base & evaluation tabs
# --------------------------------------------------------------------------- #
def render_knowledge_base(settings: Any, db: DocumentDB) -> None:
    """Browse documents, chunks and the ingestion log."""
    stats = db.get_stats()
    columns = st.columns(4)
    columns[0].metric("Documents", stats["documents"])
    columns[1].metric("Chunks", stats["chunks"])
    columns[2].metric("Tokens", f"{stats['tokens']:,}")
    columns[3].metric("Sources", stats["sources"])

    sources = db.list_sources()
    if not sources:
        st.info("The knowledge base is empty.")
        return

    selected = st.selectbox("Source", ["All sources"] + sources)
    source_filter = None if selected == "All sources" else selected
    documents = db.list_documents(source=source_filter, limit=200)
    st.markdown(f"**Documents ({len(documents)})**")
    st.dataframe(
        [
            {
                "source": document.source,
                "type": str(document.source_type),
                "page": document.page,
                "chars": document.char_count,
                "created": document.created_at,
                "preview": document.text[:160],
            }
            for document in documents
        ],
        width="stretch",
        height=240,
    )

    chunks = db.get_chunks(source=source_filter)
    st.markdown(f"**Chunks ({len(chunks)})**")
    st.dataframe(
        [
            {
                "chunk_id": chunk.chunk_id[:10],
                "source": chunk.source,
                "page": chunk.page,
                "index": chunk.chunk_index,
                "tokens": chunk.token_count,
                "language": chunk.metadata.get("language"),
                "keywords": ", ".join(chunk.metadata.get("keywords", [])[:5]),
                "preview": chunk.text[:200].replace("\n", " "),
            }
            for chunk in chunks[:300]
        ],
        width="stretch",
        height=300,
    )

    with st.expander("Ingestion log"):
        log = db.recent_ingestions(30)
        if log:
            st.dataframe(log, width="stretch")
        else:
            st.caption("No ingestion runs yet.")

    with st.expander("Graph index"):
        builder = get_index_builder(current_version())
        if builder.graph_index.exists():
            stats_graph = builder.graph_index.stats()
            st.json(stats_graph)
            triples = builder.graph_index.export_triples(limit=25)
            if triples:
                st.dataframe(triples, width="stretch")
        else:
            st.caption("Graph index not built yet.")


def render_evaluation(settings: Any) -> None:
    """Run the RAGAS / heuristic evaluation from the UI."""
    st.markdown("#### 🧪 Evaluation")
    st.caption(
        "RAGAS metrics (faithfulness, answer relevancy, context precision, context recall). "
        "Without an LLM judge the deterministic proxy metrics are used."
    )
    mode = st.radio("Mode", ["auto", "heuristic", "ragas"], horizontal=True)
    max_samples = st.slider("Questions", 1, 10, 5)
    if st.button("Run evaluation", type="primary"):
        from src.evaluation.ragas_eval import RAGASEvaluator

        with st.spinner("Evaluating…"):
            try:
                evaluator = RAGASEvaluator(
                    settings=settings,
                    pipeline=get_rag_pipeline(current_version()),
                    mode=mode,
                    max_samples=max_samples,
                )
                report = evaluator.run(with_baseline=True)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return
        st.success(f"Mode: {report.mode} · {len(report.samples)} questions")
        st.dataframe(
            [
                {
                    "metric": name,
                    "advanced": round(report.metrics.get(name, 0.0), 3),
                    "baseline": round(report.baseline_metrics.get(name, 0.0), 3) if report.baseline_metrics else None,
                    "delta": round(report.improvements().get(name, 0.0), 3) if report.baseline_metrics else None,
                }
                for name in report.metrics
            ],
            width="stretch",
        )
        with st.expander("Per-question results"):
            st.dataframe(
                [
                    {
                        "question": sample.question,
                        "answer": sample.answer[:220],
                        "citations": ", ".join(sample.citations),
                        "latency_ms": sample.latency_ms,
                    }
                    for sample in report.samples
                ],
                width="stretch",
            )
        st.download_button(
            "⬇️ Download report (JSON)",
            data=json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            file_name="ragas_report.json",
            mime="application/json",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Advanced Multi-Source RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = get_settings_cached()
    settings.ensure_directories()
    version = current_version()
    db = get_db(version)

    st.title("📚 Advanced Multi-Source RAG")
    st.caption(
        "PDF · CSV · URL ingestion → SQLite document DB → clean/chunk/tag → "
        "FAISS + sentence-window + knowledge-graph indexes → parallel retrieval → "
        "cross-encoder reranking → grounded answers with citations."
    )

    with st.sidebar:
        st.header("Knowledge base")
        render_status(settings, db)
        st.divider()
        render_ingestion(settings, db, version)
        st.divider()
        st.caption("Built with LlamaIndex · FAISS · NetworkX · Streamlit")

    chat_tab, kb_tab, eval_tab = st.tabs(["💬 Chat", "🗂️ Knowledge base", "🧪 Evaluation"])

    with chat_tab:
        pipeline = get_rag_pipeline(version)
        if pipeline.warnings:
            with st.expander("⚠️ Retrieval warnings", expanded=False):
                for warning in pipeline.warnings:
                    st.warning(warning)
        render_chat(pipeline, db)

    with kb_tab:
        render_knowledge_base(settings, db)

    with eval_tab:
        if not db.count_chunks():
            st.info("Ingest documents first.")
        else:
            render_evaluation(settings)


if __name__ == "__main__":
    main()
