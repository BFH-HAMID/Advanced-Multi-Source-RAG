"""Shared pytest fixtures for the Advanced Multi-Source RAG test-suite.

Every test runs fully offline: hashing embeddings, lexical reranking and the
extractive LLM keep the suite deterministic and fast (no model downloads, no
API keys, no network).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.config import Settings
from src.indexing.build_all import IndexBuilder
from src.models import RawDocument, SourceType
from src.pipeline import RAGPipeline, build_pipeline
from src.processing.chunker import TokenChunker
from src.processing.processor import DocumentProcessor
from src.storage.document_db import DocumentDB


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Offline, deterministic settings rooted in a temp directory."""
    return Settings(
        data_dir=tmp_path / "data",
        embedding_backend="hashing",
        embedding_dim=256,
        embedding_cache=False,
        reranker_backend="lexical",
        llm_provider="extractive",
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        min_chunk_chars=20,
        csv_rows_per_document=3,
        graph_max_hops=2,
        sentence_window_size=1,
        retrieval_top_k=6,
        rerank_top_n=3,
        max_context_tokens=1200,
    )


@pytest.fixture()
def sample_documents() -> list[RawDocument]:
    """A small bilingual enterprise corpus."""
    return [
        RawDocument(
            source="leave_policy.txt",
            source_type=SourceType.TEXT,
            text=(
                "Acme Bangladesh Ltd is headquartered in Dhaka. "
                "Every permanent employee is entitled to 20 days of annual leave per calendar year. "
                "Employees also receive 10 days of sick leave. "
                "Annual leave requests must be submitted to People Operations 14 days in advance. "
                "Core working hours are 10:00 to 16:00. "
                "Employees may work remotely up to 3 days per week. "
                "Expenses above BDT 10,000 require approval from the reporting manager. "
                "The annual learning budget is BDT 50,000 per employee."
            ),
            title="leave_policy",
            metadata={"loader": "test"},
        ),
        RawDocument(
            source="hr_bangla.txt",
            source_type=SourceType.TEXT,
            text=(
                "আসমে বাংলাদেশ লিমিটেড ২০১৬ সালে ঢাকায় প্রতিষ্ঠিত হয়। "
                "প্রতি ক্যালেন্ডার বছরে প্রত্যেক স্থায়ী কর্মী ২০ দিন বার্ষিক ছুটি পাবেন। "
                "কর্মীরা সপ্তাহে সর্বোচ্চ ৩ দিন রিমোট কাজ করতে পারেন। "
                "প্রতি কর্মীর জন্য বছরে ৫০,০০০ টাকার শিক্ষা বাজেট রয়েছে। "
                "নিরাপত্তা ঘটনা ২৪ ঘণ্টার মধ্যে জানাতে হবে।"
            ),
            title="hr_bangla",
            metadata={"loader": "test"},
        ),
        RawDocument(
            source="employees.csv",
            source_type=SourceType.CSV,
            text=(
                "Table: employees.csv (rows 1-3 of 3)\n"
                "Columns: name, department, location\n"
                "Row 1: name: Nusrat Jahan | department: People Operations | location: Dhaka.\n"
                "Row 2: name: Sadia Islam | department: Engineering | location: Chattogram.\n"
                "Row 3: name: Mahmudul Karim | department: Security | location: Dhaka."
            ),
            page=1,
            title="employees",
            metadata={"loader": "csv"},
        ),
    ]


@pytest.fixture()
def db(settings: Settings) -> DocumentDB:
    """Empty document database."""
    return DocumentDB(settings=settings)


@pytest.fixture()
def populated_db(settings: Settings, sample_documents: list[RawDocument]) -> DocumentDB:
    """Document database containing the sample corpus."""
    database = DocumentDB(settings=settings)
    processor = DocumentProcessor(settings=settings)
    chunks, _ = processor.process(sample_documents)
    database.insert_documents(sample_documents)
    database.insert_chunks(chunks)
    return database


@pytest.fixture()
def built_indexes(settings: Settings, populated_db: DocumentDB) -> IndexBuilder:
    """Vector, sentence-window and graph indexes built from the sample corpus."""
    builder = IndexBuilder(settings=settings, db=populated_db)
    builder.build_all()
    return builder


@pytest.fixture()
def pipeline(settings: Settings, built_indexes: IndexBuilder) -> RAGPipeline:
    """End-to-end pipeline wired to the built indexes."""
    return build_pipeline(settings=settings, db=built_indexes.db)


@pytest.fixture()
def chunker(settings: Settings) -> TokenChunker:
    """Chunker using the test settings (120/20 tokens)."""
    return TokenChunker(settings=settings)


@pytest.fixture()
def pdf_factory(tmp_path: Path):
    """Create small PDFs with PyMuPDF for loader tests."""

    def _create(pages: list[str], name: str = "sample.pdf") -> Path:
        import pymupdf

        document = pymupdf.open()
        for page_text in pages:
            page = document.new_page()
            page.insert_text((72, 72), page_text, fontsize=11, fontname="helv")
        target = tmp_path / name
        document.save(target)
        document.close()
        return target

    return _create


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove provider credentials so fallbacks are exercised deterministically."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield
