"""Tests for cleaning, metadata tagging and the document database."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.models import Chunk, RawDocument, SourceType
from src.processing.cleaner import TextCleaner, clean_text
from src.processing.metadata import MetadataTagger, build_node_metadata
from src.processing.processor import DocumentProcessor
from src.storage.document_db import DocumentDB


# --------------------------------------------------------------------------- #
# Cleaner
# --------------------------------------------------------------------------- #
def test_cleaner_removes_noise_and_page_numbers(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    text = "Page 4 of 9\n\nThe leave policy grants 20 days.   \n\n- 12  \n"
    cleaned = cleaner.clean(text)
    assert "Page 4" not in cleaned
    assert "leave policy" in cleaned
    assert "  " not in cleaned


def test_cleaner_repairs_hyphenation(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    assert "multilingual" in cleaner.clean("A multi-\nlingual knowledge base.")


def test_cleaner_deduplicates_paragraphs(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    cleaned = cleaner.clean("Unique paragraph.\n\nUnique paragraph.\n\nAnother one.")
    assert cleaned.lower().count("unique paragraph") == 1


def test_cleaner_removes_repeated_headers(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    pages = [
        "Acme Confidential\nLeave policy is 20 days.\nPage 1 of 3",
        "Acme Confidential\nRemote work is 3 days.\nPage 2 of 3",
        "Acme Confidential\nLearning budget is BDT 50,000.\nPage 3 of 3",
    ]
    cleaned = cleaner.remove_repeated_boilerplate(pages)
    assert all("Acme Confidential" not in page for page in cleaned)
    assert "Leave policy" in cleaned[0]


def test_cleaner_strips_markdown(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    cleaned = cleaner.clean("## Heading\n**bold** and [link](https://x.example)", strip_markdown=True)
    assert "#" not in cleaned and "**" not in cleaned
    assert "link" in cleaned and "https" not in cleaned


def test_clean_text_helper(settings: Settings) -> None:
    assert "policy" in clean_text("  The  policy  text. ", settings)


def test_cleaner_keeps_bangla_text(settings: Settings) -> None:
    cleaner = TextCleaner(settings)
    cleaned = cleaner.clean("পৃষ্ঠা ৩\n\nবার্ষিক ছুটি ২০ দিন।")
    assert "বার্ষিক ছুটি" in cleaned


# --------------------------------------------------------------------------- #
# Metadata tagging
# --------------------------------------------------------------------------- #
def test_language_detection() -> None:
    assert MetadataTagger.detect_language("This is an English policy document.") == "en"
    assert MetadataTagger.detect_language("এটি একটি বাংলা নীতিমালা নথি।") == "bn"
    assert MetadataTagger.detect_language("Acme Bangladesh বার্ষিক ছুটি policy নথি।") == "mixed"


def test_keyword_extraction() -> None:
    keywords = MetadataTagger.extract_keywords(
        "Annual leave policy: annual leave is 20 days per year, leave requests via portal."
    )
    assert "leave" in keywords
    assert "the" not in keywords


def test_section_detection() -> None:
    assert MetadataTagger.extract_section("Leave and Absence Policy\nEmployees get 20 days.") == (
        "Leave and Absence Policy"
    )
    assert MetadataTagger.extract_section("This is a full sentence.") is None


def test_tag_chunk_adds_metadata(settings: Settings) -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Annual leave is 20 days per year.",
        chunk_index=0,
        source="policy.pdf",
        source_type=SourceType.PDF,
        page=2,
    )
    tagged = MetadataTagger(settings).tag([chunk])[0]
    assert tagged.metadata["language"] == "en"
    assert tagged.metadata["citation"] == "policy.pdf, p.2"
    assert "leave" in tagged.metadata["keywords"]
    assert tagged.token_count > 0


def test_build_node_metadata_flattens_values(settings: Settings) -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Some text",
        chunk_index=3,
        source="policy.pdf",
        source_type=SourceType.PDF,
        page=None,
        title="Policy",
        metadata={"keywords": ["leave", "policy"], "section": None, "window": "reserved"},
    )
    metadata = build_node_metadata(chunk)
    assert metadata["page"] == -1
    assert metadata["keywords"] == "leave, policy"
    assert "section" not in metadata
    assert "window" not in metadata
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
def test_processor_produces_tagged_chunks(settings: Settings, sample_documents) -> None:
    chunks, stats = DocumentProcessor(settings=settings).process(sample_documents)
    assert stats.documents_in == 3
    assert stats.chunks == len(chunks) > 0
    assert all(chunk.metadata.get("language") for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_processor_deduplicates_identical_chunks(settings: Settings) -> None:
    document = RawDocument(
        source="dup.txt",
        source_type=SourceType.TEXT,
        text="Same sentence repeated. Same sentence repeated.",
    )
    chunks, _ = DocumentProcessor(settings=settings).process([document])
    assert len({chunk.dedupe_key for chunk in chunks}) == len(chunks)


def test_reprocess_from_db(settings: Settings, populated_db: DocumentDB) -> None:
    before = populated_db.count_chunks()
    chunks, _ = DocumentProcessor(settings=settings).reprocess_from_db(populated_db)
    assert len(chunks) == populated_db.count_chunks()
    assert len(chunks) == before


# --------------------------------------------------------------------------- #
# Document DB
# --------------------------------------------------------------------------- #
def test_database_insert_and_read(settings: Settings, sample_documents) -> None:
    db = DocumentDB(settings=settings)
    inserted = db.insert_documents(sample_documents)
    assert inserted == len(sample_documents)
    assert db.count_documents() == len(sample_documents)

    stored = db.get_document(sample_documents[0].doc_id)
    assert stored is not None
    assert stored.text == sample_documents[0].text
    assert stored.source_type is sample_documents[0].source_type


def test_database_deduplicates_identical_documents(settings: Settings, sample_documents) -> None:
    db = DocumentDB(settings=settings)
    db.insert_documents(sample_documents)
    assert db.insert_documents(sample_documents) == 0


def test_database_chunks_roundtrip(settings: Settings, populated_db: DocumentDB) -> None:
    chunks = populated_db.get_chunks()
    assert chunks
    fetched = populated_db.get_chunks_by_ids([chunk.chunk_id for chunk in chunks[:2]])
    assert len(fetched) == 2
    assert populated_db.count_chunks() == len(chunks)


def test_database_filters_and_sources(settings: Settings, populated_db: DocumentDB) -> None:
    assert "employees.csv" in populated_db.list_sources()
    csv_documents = populated_db.list_documents(source_type="csv")
    assert csv_documents and all(str(document.source_type) == "csv" for document in csv_documents)


def test_database_delete_source_and_clear(settings: Settings, populated_db: DocumentDB) -> None:
    populated_db.delete_source("employees.csv")
    assert "employees.csv" not in populated_db.list_sources()
    populated_db.clear_all()
    assert populated_db.count_documents() == 0
    assert populated_db.count_chunks() == 0


def test_database_ingestion_log_and_meta(settings: Settings) -> None:
    db = DocumentDB(settings=settings)
    db.log_ingestion("a.pdf", "pdf", "ok", "", 2, 4)
    log = db.recent_ingestions()
    assert log and log[0]["source"] == "a.pdf"

    db.set_meta("manifest", {"chunks": 4})
    assert db.get_meta("manifest") == {"chunks": 4}
    assert db.get_meta("missing", "default") == "default"


def test_database_stats(settings: Settings, populated_db: DocumentDB) -> None:
    stats = populated_db.get_stats()
    assert stats["documents"] == 3
    assert stats["chunks"] > 0
    assert stats["documents_by_type"]["text"] == 2
    assert Path(stats["db_path"]).exists()
