"""Tests for the PDF / CSV / URL / text document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.errors import LoaderError
from src.ingestion.loaders import (
    CSVLoader,
    LoaderRegistry,
    PDFLoader,
    TextLoader,
    URLLoader,
    load_path,
)
from src.models import SourceType


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def test_pdf_loader_extracts_pages(settings: Settings, pdf_factory) -> None:
    pdf = pdf_factory(["Annual leave is 20 days.", "Core hours are 10 to 16."])
    documents = PDFLoader(settings).load(pdf)

    assert len(documents) == 2
    assert documents[0].page == 1
    assert documents[1].page == 2
    assert documents[0].source_type is SourceType.PDF
    assert "Annual leave" in documents[0].text
    assert documents[0].metadata["page_count"] == 2


def test_pdf_loader_skips_blank_pages(settings: Settings, pdf_factory) -> None:
    pdf = pdf_factory(["Only page with text."])
    documents = PDFLoader(settings).load(pdf)
    assert len(documents) == 1
    assert documents[0].text.strip()


def test_pdf_loader_missing_file(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(LoaderError, match="does not exist"):
        PDFLoader(settings).load(tmp_path / "nope.pdf")


def test_pdf_loader_rejects_corrupt_file(settings: Settings, tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    with pytest.raises(LoaderError):
        PDFLoader(settings).load(broken)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_csv_loader_builds_row_blocks(settings: Settings, tmp_path: Path) -> None:
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "name,department,location\n"
        "Nusrat Jahan,People Operations,Dhaka\n"
        "Sadia Islam,Engineering,Chattogram\n"
        "Arif Mahmud,Support,Chattogram\n",
        encoding="utf-8",
    )
    documents = CSVLoader(settings).load(csv_path)  # 3 rows per document in tests

    assert len(documents) == 1
    document = documents[0]
    assert document.source_type is SourceType.CSV
    assert document.page == 1
    assert "Nusrat Jahan" in document.text
    assert "Columns: name, department, location" in document.text
    assert document.metadata["total_rows"] == 3
    assert document.text.rstrip().endswith(".")


def test_csv_loader_handles_bangla_utf8(settings: Settings, tmp_path: Path) -> None:
    csv_path = tmp_path / "kormi.csv"
    csv_path.write_text(
        "নাম,বিভাগ,শহর\n"
        "নুসরাত জাহান,মানবসম্পদ,ঢাকা\n"
        "সাদিয়া ইসলাম,ইঞ্জিনিয়ারিং,চট্টগ্রাম\n",
        encoding="utf-8",
    )
    documents = CSVLoader(settings).load(csv_path)
    assert len(documents) == 1
    assert "নুসরাত জাহান" in documents[0].text
    assert documents[0].metadata["columns"] == ["নাম", "বিভাগ", "শহর"]


def test_csv_loader_splits_large_tables(settings: Settings, tmp_path: Path) -> None:
    rows = "\n".join(f"name{i},dept{i},Dhaka" for i in range(7))
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("name,department,location\n" + rows + "\n", encoding="utf-8")
    documents = CSVLoader(settings).load(csv_path)
    assert len(documents) == 3  # 7 rows / 3 per block
    assert [document.page for document in documents] == [1, 2, 3]


def test_csv_loader_rejects_empty_file(settings: Settings, tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(LoaderError):
        CSVLoader(settings).load(empty)


# --------------------------------------------------------------------------- #
# URL
# --------------------------------------------------------------------------- #
HTML = """
<html><head><title>Leave policy | Acme</title></head><body>
<nav>Home | About | Contact</nav>
<article>
  <h1>Leave policy</h1>
  <p>Every permanent employee is entitled to 20 days of annual leave per calendar year.</p>
  <p>Requests must be submitted to People Operations at least 14 days in advance.</p>
  <p>Employees may work remotely up to 3 days per week, subject to team coverage.</p>
</article>
<footer>© Acme Bangladesh Ltd</footer>
</body></html>
"""


def test_url_loader_extracts_main_content(settings: Settings) -> None:
    documents = URLLoader(settings).load_html(HTML, url="https://acme.example/hr/leave")
    assert len(documents) == 1
    document = documents[0]
    assert document.source_type is SourceType.URL
    assert "20 days of annual leave" in document.text
    assert document.page is None
    assert document.metadata["host"] == "acme.example"
    assert "Home | About" not in document.text


def test_url_loader_rejects_invalid_url(settings: Settings) -> None:
    with pytest.raises(LoaderError, match="invalid URL"):
        URLLoader(settings).load("not-a-url")


def test_url_loader_empty_html(settings: Settings) -> None:
    with pytest.raises(LoaderError):
        URLLoader(settings).load_html("   ", url="https://acme.example/empty")


# --------------------------------------------------------------------------- #
# Text + registry
# --------------------------------------------------------------------------- #
def test_text_loader_reads_markdown(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nRemote work is allowed 3 days per week.", encoding="utf-8")
    documents = TextLoader(settings).load(path)
    assert len(documents) == 1
    assert documents[0].source_type is SourceType.TEXT
    assert "Remote work" in documents[0].text


def test_registry_dispatches_by_suffix(settings: Settings, tmp_path: Path, pdf_factory) -> None:
    registry = LoaderRegistry(settings)
    assert ".pdf" in registry.supported_suffixes
    assert registry.get_loader(tmp_path / "a.pdf").source_type is SourceType.PDF
    assert registry.get_loader(tmp_path / "a.csv").source_type is SourceType.CSV
    with pytest.raises(LoaderError, match="unsupported file type"):
        registry.get_loader(tmp_path / "a.exe")


def test_registry_collects_errors_without_aborting(settings: Settings, tmp_path: Path, pdf_factory) -> None:
    good = tmp_path / "good.txt"
    good.write_text("Policy text that is long enough to be ingested properly.", encoding="utf-8")
    documents, errors = LoaderRegistry(settings).load_many([good, tmp_path / "missing.pdf"])
    assert len(documents) == 1
    assert len(errors) == 1


def test_load_path_helper(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "helper.txt"
    path.write_text("A short document about leave policy and remote work.", encoding="utf-8")
    assert load_path(path, settings)[0].source == "helper.txt"
