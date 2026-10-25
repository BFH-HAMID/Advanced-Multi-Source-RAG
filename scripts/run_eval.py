"""CLI: evaluate the RAG pipeline with RAGAS (or the offline proxy metrics).

Examples:
    python scripts/run_eval.py                        # auto: RAGAS if an LLM is configured
    python scripts/run_eval.py --mode heuristic       # no LLM needed
    python scripts/run_eval.py --no-baseline --max-samples 5
    python scripts/run_eval.py --testset data/evaluation/sample_testset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, get_settings  # noqa: E402
from src.evaluation.ragas_eval import DEFAULT_TESTSET, RAGASEvaluator  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--mode", choices=["auto", "ragas", "heuristic"], default="auto")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET), help="Path to the test set JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of questions")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the vector-only baseline")
    parser.add_argument("--output-dir", default=None, help="Directory for JSON/markdown reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    pipeline = build_pipeline(settings)
    description = pipeline.describe()
    print("Configuration:")
    print(json.dumps(description, indent=2, ensure_ascii=False))

    evaluator = RAGASEvaluator(
        settings=settings,
        pipeline=pipeline,
        testset_path=args.testset,
        mode=args.mode,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
    report = evaluator.run(with_baseline=not args.no_baseline)

    print("\n" + "=" * 78)
    print(report.to_markdown())
    print("=" * 78)
    print("Metrics:", json.dumps(report.to_dict()["metrics"], indent=2))
    if report.baseline_metrics:
        print("Baseline:", json.dumps(report.to_dict()["baseline_metrics"], indent=2))
        print("Delta:", json.dumps(report.improvements(), indent=2))
    for kind, path in report.output_paths.items():
        print(f"• {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
