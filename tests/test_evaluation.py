"""Tests for the ingestion pipeline and the evaluation module (offline mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.errors import EvaluationError
from src.evaluation.ragas_eval import RAGASEvaluator
from src.ingestion.pipeline import IngestionPipeline, safe_filename
from src.storage.document_db import DocumentDB


# --------------------------------------------------------------------------- #
# Ingestion pipeline
# --------------------------------------------------------------------------- #
def test_safe_filename_sanitises_paths() -> None:
    assert safe_filename("../../etc/passwd") == "passwd"
    assert "/" not in safe_filename("dir/file name.csv")
    assert safe_filename("বাংলা নীতি.txt").endswith(".txt")


def test_ingest_files_creates_documents_and_chunks(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text(
        "Annual leave is 20 days. Remote work is allowed 3 days per week. " * 3,
        encoding="utf-8",
    )
    db = DocumentDB(settings=settings)
    report = IngestionPipeline(settings=settings, db=db).ingest_files([path])

    assert report.ok
    assert report.documents == 1
    assert report.new_documents == 1
    assert report.chunks >= 1
    assert db.count_documents() == 1
    assert db.count_chunks() == report.chunks


def test_ingest_missing_file_reports_error(settings: Settings, tmp_path: Path) -> None:
    db = DocumentDB(settings=settings)
    report = IngestionPipeline(settings=settings, db=db).ingest_files([tmp_path / "missing.pdf"])
    assert not report.ok
    assert report.errors
    assert report.details[0].status == "error"


def test_ingest_text_and_replacement(settings: Settings) -> None:
    db = DocumentDB(settings=settings)
    pipeline = IngestionPipeline(settings=settings, db=db)

    first = pipeline.ingest_text("Policy version one about leave.", name="note.txt")
    assert first.chunks == 1

    second = pipeline.ingest_text("Policy version two about remote work and leave.", name="note.txt")
    assert second.chunks >= 1
    assert db.count_documents() == 1  # replaced, not duplicated


def test_save_upload_writes_file(settings: Settings) -> None:
    pipeline = IngestionPipeline(settings=settings)
    path = pipeline.save_upload("upload.csv", b"a,b\n1,2\n")
    assert path.exists()
    assert path.name == "upload.csv"


def test_ingest_urls_invalid_url(settings: Settings) -> None:
    report = IngestionPipeline(settings=settings).ingest_urls(["ftp://example.com/file"])
    assert report.errors
    assert report.details[0].status == "error"


def test_ingest_sample_sources(settings: Settings, tmp_path: Path) -> None:
    csv_path = tmp_path / "team.csv"
    csv_path.write_text(
        "name,department\nNusrat Jahan,People Operations\nSadia Islam,Engineering\n",
        encoding="utf-8",
    )
    db = DocumentDB(settings=settings)
    report = IngestionPipeline(settings=settings, db=db).ingest_files([csv_path])
    assert report.ok
    assert report.details[0].source == "team.csv"
    assert db.get_stats()["documents_by_type"]["csv"] == 1


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_evaluator_heuristic_mode_reports_metrics(settings: Settings, pipeline, tmp_path: Path) -> None:
    testset = tmp_path / "testset.json"
    testset.write_text(
        json.dumps(
            [
                {
                    "question": "How many days of annual leave do employees get?",
                    "ground_truth": "Every permanent employee is entitled to 20 days of annual leave.",
                },
                {
                    "question": "What is the capital of France?",
                    "ground_truth": "Not in the documents.",
                    "unanswerable": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    evaluator = RAGASEvaluator(
        settings=settings,
        pipeline=pipeline,
        testset_path=testset,
        mode="heuristic",
        output_dir=tmp_path / "reports",
    )
    report = evaluator.run(with_baseline=True)

    assert report.mode == "heuristic"
    assert len(report.samples) == 2
    assert set(report.metrics) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert all(0.0 <= value <= 1.0 for value in report.metrics.values())
    assert report.baseline_metrics
    assert report.extra["refusal_accuracy"] in (0.0, 1.0)
    assert Path(report.output_paths["json"]).exists()
    assert "RAG evaluation report" in Path(report.output_paths["markdown"]).read_text(encoding="utf-8")


def test_evaluator_loads_default_testset(settings: Settings, pipeline) -> None:
    evaluator = RAGASEvaluator(settings=settings, pipeline=pipeline)
    questions = evaluator.load_testset(Path("data/evaluation/sample_testset.json"))
    assert len(questions) >= 5
    assert {"question", "ground_truth"} <= set(questions[0])


def test_evaluator_rejects_missing_testset(settings: Settings, pipeline, tmp_path: Path) -> None:
    evaluator = RAGASEvaluator(settings=settings, pipeline=pipeline, testset_path=tmp_path / "nope.json")
    with pytest.raises(EvaluationError):
        evaluator.load_testset()


def test_evaluator_ragas_mode_requires_llm(settings: Settings, pipeline) -> None:
    evaluator = RAGASEvaluator(settings=settings, pipeline=pipeline, mode="ragas")
    assert pipeline.llm.provider == "extractive"
    with pytest.raises(EvaluationError, match="real LLM judge"):
        evaluator.resolve_mode()


def test_report_serialisation(settings: Settings, pipeline, tmp_path: Path) -> None:
    testset = tmp_path / "testset.json"
    testset.write_text(
        json.dumps([{"question": "remote work policy", "ground_truth": "3 days per week"}]),
        encoding="utf-8",
    )
    report = RAGASEvaluator(
        settings=settings,
        pipeline=pipeline,
        testset_path=testset,
        mode="heuristic",
        output_dir=tmp_path,
    ).run(with_baseline=False)
    payload = report.to_dict()
    assert json.dumps(payload)
    assert payload["metrics"]
    assert report.average_latency_ms >= 0
