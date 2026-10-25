"""CLI: ingest sources and build every index.

Examples:
    python scripts/build_index.py                          # ingest samples + build
    python scripts/build_index.py --files data/raw/*.pdf --no-samples
    python scripts/build_index.py --reprocess              # re-chunk from SQLite
    python scripts/build_index.py --skip-graph             # fast vector-only build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, get_settings  # noqa: E402
from src.indexing.build_all import IndexBuilder  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.processing.processor import DocumentProcessor  # noqa: E402
from src.storage.document_db import DocumentDB  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents and build the RAG indexes")
    parser.add_argument("--files", nargs="*", default=[], help="Files to ingest (PDF/CSV/TXT/MD)")
    parser.add_argument("--urls", nargs="*", default=[], help="URLs to ingest")
    parser.add_argument("--no-samples", action="store_true", help="Do not ingest data/samples")
    parser.add_argument("--reset", action="store_true", help="Clear the document database first")
    parser.add_argument("--reprocess", action="store_true", help="Re-chunk existing documents from SQLite")
    parser.add_argument("--skip-graph", action="store_true", help="Skip the graph index")
    parser.add_argument("--skip-window", action="store_true", help="Skip the sentence-window index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()
    db = DocumentDB(settings=settings)

    if args.reset:
        db.clear_all()
        print("• Cleared the document database")

    pipeline = IngestionPipeline(settings=settings, db=db)
    files = [Path(path) for path in args.files]
    if not args.no_samples and not files and not args.urls:
        files = sorted((settings.data_dir / "samples").iterdir())

    if args.reprocess:
        chunks, stats = DocumentProcessor(settings=settings).reprocess_from_db(db)
        print(f"• Re-processed {stats.documents_in} documents into {len(chunks)} chunks")
    else:
        if files:
            report = pipeline.ingest_files(files)
            print(
                f"• Ingested {len(report.sources)} file(s): {report.documents} documents, "
                f"{report.chunks} chunks"
            )
            for error in report.errors:
                print(f"  ! {error}")
        if args.urls:
            report = pipeline.ingest_urls(args.urls)
            print(f"• Ingested {len(report.sources)} URL(s): {report.chunks} chunks")
            for error in report.errors:
                print(f"  ! {error}")

    total_chunks = db.count_chunks()
    if total_chunks == 0:
        print("No chunks available; nothing to index.")
        return 1

    builder = IndexBuilder(settings=settings, db=db)
    last_stage = {"stage": "", "done": 0, "total": 0}

    def progress(stage: str, done: int, total: int) -> None:
        if (stage, done, total) != (last_stage["stage"], last_stage["done"], last_stage["total"]):
            last_stage.update(stage=stage, done=done, total=total)
            print(f"  … {stage}: {done}/{total}")

    print(f"• Building indexes for {total_chunks} chunks")
    report = builder.build_all(
        skip_graph=args.skip_graph,
        skip_window=args.skip_window,
        progress=progress,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    if report.errors:
        print("Completed with warnings:")
        for error in report.errors:
            print(f"  ! {error}")
    print(f"• Indexes written to {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
