"""CLI: ask a question against the indexed knowledge base.

Examples:
    python scripts/ask.py "How many annual leave days do employees get?"
    python scripts/ask.py "বার্ষিক ছুটি কত দিন?" --json
    python scripts/ask.py "What is the SLA for Nimbus OCR?" --show-contexts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_logging, get_settings  # noqa: E402
from src.generation.formatter import render_markdown  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the RAG knowledge base")
    parser.add_argument("question", help="Question (English or Bangla)")
    parser.add_argument("--json", action="store_true", help="Print the full JSON answer object")
    parser.add_argument("--show-contexts", action="store_true", help="Print the retrieved contexts")
    parser.add_argument("--top-k", type=int, default=None, help="Fused candidates before reranking")
    parser.add_argument("--rerank-top-n", type=int, default=None, help="Contexts kept after reranking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("WARNING")
    settings = get_settings()
    pipeline = build_pipeline(settings)
    pipeline.top_k = args.top_k or pipeline.top_k
    pipeline.rerank_top_n = args.rerank_top_n or pipeline.rerank_top_n

    try:
        answer = pipeline.answer(args.question)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(render_markdown(answer))
    if args.show_contexts:
        print("\n--- contexts ---")
        for context in answer.contexts:
            print(
                f"[{context.rank + 1}] {context.citation_key} "
                f"(score={float(context.rerank_score or 0):.3f}, via {', '.join(context.retrievers)})"
            )
            print(context.text.strip()[:500])
            print()
    print(f"\n({answer.timings_ms.get('total_ms', 0):.0f} ms · {answer.provider}/{answer.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
