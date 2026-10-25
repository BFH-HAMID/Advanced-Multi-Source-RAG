"""RAGAS evaluation with an offline fallback.

Metrics (RAGAS >= 0.2):
    * ``faithfulness`` – is every answer statement supported by the context?
    * ``answer_relevancy`` – does the answer address the question?
    * ``context_precision`` – are the retrieved contexts relevant/well ranked?
    * ``context_recall`` (bonus) – did retrieval cover the ground truth?

Because RAGAS itself needs an LLM judge, the script automatically falls back to
a deterministic **heuristic proxy** (embedding similarity based) when no real
LLM is configured — so `python scripts/run_eval.py` always produces a report.

RAGAS is wired to the project's own LLM and embedding models through thin
adapters, so Claude, OpenAI, Ollama and bge-m3 all work without extra
LangChain model packages.

Example:
    >>> from src.evaluation.ragas_eval import RAGASEvaluator   # doctest: +SKIP
    >>> report = RAGASEvaluator().run()                        # doctest: +SKIP
    >>> report.mode in {"ragas", "heuristic"}                  # doctest: +SKIP
    True
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.errors import EvaluationError
from src.generation.llm import LLMClient
from src.models import utc_now_iso
from src.pipeline import RAGPipeline, build_pipeline
from src.rerank.reranker import Reranker
from src.retrieval import MultiRetriever, VectorRetriever
from src.text_utils import normalise_whitespace, split_sentences, tokenize_words

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DEFAULT_TESTSET = Path("data/evaluation/sample_testset.json")


# --------------------------------------------------------------------------- #
# RAGAS adapters (project LLM/embeddings -> ragas interfaces)
# --------------------------------------------------------------------------- #
def _build_ragas_llm(llm: LLMClient, run_config: Any = None) -> Any:
    """Wrap the project's :class:`LLMClient` as a RAGAS LLM."""
    try:
        from langchain_core.outputs import Generation, LLMResult
        from ragas.llms.base import BaseRagasLLM
    except ImportError as exc:  # pragma: no cover - ragas not installed
        raise EvaluationError(f"ragas is not installed: {exc}") from exc

    class ProjectRagasLLM(BaseRagasLLM):  # type: ignore[misc]
        """Adapter that lets RAGAS call our provider-agnostic LLM client."""

        client: Any = None

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            outputs = [self.client.complete(text, temperature=temperature) for _ in range(max(1, n))]
            return LLMResult(generations=[[Generation(text=output)] for output in outputs])

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                temperature if temperature is not None else 1e-8,
                stop,
                callbacks,
            )

    adapter = ProjectRagasLLM(client=llm)
    if run_config is not None:
        adapter.set_run_config(run_config)
    return adapter


def _build_ragas_embeddings(embed_model: Any, run_config: Any = None) -> Any:
    """Wrap a LlamaIndex embedding model as RAGAS embeddings."""
    try:
        from ragas.embeddings.base import BaseRagasEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise EvaluationError(f"ragas is not installed: {exc}") from exc

    class ProjectRagasEmbeddings(BaseRagasEmbeddings):  # type: ignore[misc]
        """Adapter around a LlamaIndex ``BaseEmbedding``."""

        model: Any = None

        def embed_query(self, text: str) -> list[float]:
            return list(self.model.get_query_embedding(text))

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [list(vector) for vector in self.model.get_text_embedding_batch(list(texts))]

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    adapter = ProjectRagasEmbeddings(model=embed_model)
    if run_config is not None:
        adapter.set_run_config(run_config)
    return adapter


# --------------------------------------------------------------------------- #
# Report models
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class EvalSample:
    """One evaluated question."""

    question: str
    ground_truth: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    grounded: bool = True
    unanswerable: bool = False
    language: str = "en"
    per_sample_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationReport:
    """Aggregate evaluation result."""

    mode: str
    metrics: dict[str, float]
    samples: list[EvalSample]
    config: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    output_paths: dict[str, str] = field(default_factory=dict)

    @property
    def average_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.latency_ms for sample in self.samples) / len(self.samples)

    def improvements(self) -> dict[str, float]:
        """Metric deltas between the advanced pipeline and the baseline."""
        if not self.baseline_metrics:
            return {}
        deltas: dict[str, float] = {}
        for name, value in self.metrics.items():
            baseline = self.baseline_metrics.get(name)
            if baseline is not None:
                deltas[name] = round(value - baseline, 4)
        return deltas

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "baseline_metrics": {key: round(value, 4) for key, value in self.baseline_metrics.items()},
            "improvements": self.improvements(),
            "config": self.config,
            "extra": self.extra,
            "created_at": self.created_at,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "samples": [sample.to_dict() for sample in self.samples],
            "output_paths": self.output_paths,
        }

    def to_markdown(self) -> str:
        """Render a human readable report (also used as the CLI summary)."""
        lines = [
            "# RAG evaluation report",
            "",
            f"- **Mode:** {self.mode}",
            f"- **Created:** {self.created_at}",
            f"- **Questions:** {len(self.samples)}",
            f"- **Average latency:** {self.average_latency_ms:.0f} ms",
            "",
            "## Metrics",
            "",
            "| Metric | Advanced pipeline | Vector baseline | Δ |",
            "| --- | --- | --- | --- |",
        ]
        for name in METRIC_NAMES:
            if name not in self.metrics and name not in self.baseline_metrics:
                continue
            advanced = self.metrics.get(name)
            baseline = self.baseline_metrics.get(name)
            delta = self.improvements().get(name)
            lines.append(
                "| {name} | {adv} | {base} | {delta} |".format(
                    name=name,
                    adv=f"{advanced:.3f}" if advanced is not None else "–",
                    base=f"{baseline:.3f}" if baseline is not None else "–",
                    delta=f"{delta:+.3f}" if delta is not None else "–",
                )
            )
        if self.extra:
            lines += ["", "## Details", ""]
            for key, value in self.extra.items():
                lines.append(f"- **{key}:** {value}")
        lines += ["", "## Per-question results", "", "| # | Question | Answer (truncated) | Citations |", "| --- | --- | --- | --- |"]
        for position, sample in enumerate(self.samples, start=1):
            answer = normalise_whitespace(sample.answer)[:160].replace("|", "\\|")
            citations = ", ".join(sample.citations[:3]) or "–"
            question = sample.question.replace("|", "\\|")
            lines.append(f"| {position} | {question} | {answer} | {citations} |")
        lines += ["", f"```json\n{json.dumps(self.config, ensure_ascii=False, indent=2)}\n```", ""]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class RAGASEvaluator:
    """Run the test set through the pipeline and score it."""

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: RAGPipeline | None = None,
        testset_path: Path | str | None = None,
        mode: str = "auto",
        max_samples: int | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        """
        Args:
            settings: Project settings.
            pipeline: Pre-built pipeline (default: :func:`build_pipeline`).
            testset_path: JSON test set (question/ground_truth/contexts).
            mode: ``auto`` (RAGAS when an LLM is configured), ``ragas`` or ``heuristic``.
            max_samples: Limit the number of questions (useful for smoke tests).
            output_dir: Where to write JSON/markdown reports.
        """
        self.settings = settings or get_settings()
        self.pipeline = pipeline or build_pipeline(self.settings)
        self.testset_path = Path(testset_path or DEFAULT_TESTSET)
        self.mode = mode.lower()
        self.max_samples = max_samples
        self.output_dir = Path(output_dir or self.settings.eval_dir)
        self._embed_model: Any = None

    # ------------------------------------------------------------- utilities
    @property
    def embed_model(self) -> Any:
        """Shared embedding model used by RAGAS and the heuristic metrics."""
        if self._embed_model is None:
            from src.indexing.embeddings import get_embed_model

            self._embed_model = get_embed_model(self.settings)
        return self._embed_model

    def resolve_mode(self) -> str:
        """Decide whether real RAGAS metrics can be computed."""
        if self.mode == "heuristic":
            return "heuristic"
        provider = self.pipeline.llm.provider
        available = self.pipeline.llm.is_available and provider != "extractive"
        if self.mode == "ragas" and not available:
            raise EvaluationError(
                "RAGAS needs a real LLM judge; configure OPENAI_API_KEY/ANTHROPIC_API_KEY "
                "or OLLAMA and set LLM_PROVIDER accordingly, or use --mode heuristic"
            )
        if not available:
            logger.warning(
                "No real LLM available (provider=%s); using heuristic proxy metrics. "
                "Configure an LLM provider to run full RAGAS metrics.",
                provider,
            )
            return "heuristic"
        return "ragas"

    def load_testset(self, path: Path | str | None = None) -> list[dict[str, Any]]:
        """Load the evaluation questions."""
        target = Path(path or self.testset_path)
        if not target.exists():
            raise EvaluationError(
                f"test set not found: {target}. Run `python scripts/generate_samples.py` first."
            )
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid test set JSON: {exc}") from exc
        rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise EvaluationError("test set must be a non-empty list of questions")
        cleaned = [
            row
            for row in rows
            if isinstance(row, dict) and str(row.get("question", "")).strip()
        ]
        if self.max_samples:
            cleaned = cleaned[: self.max_samples]
        logger.info("Loaded %s evaluation questions from %s", len(cleaned), target)
        return cleaned

    # ------------------------------------------------------------- pipeline
    def collect(
        self,
        questions: Sequence[dict[str, Any]],
        pipeline: RAGPipeline | None = None,
    ) -> list[EvalSample]:
        """Answer every question and capture answer + contexts."""
        active = pipeline or self.pipeline
        samples: list[EvalSample] = []
        for row in questions:
            question = str(row["question"])
            started = time.perf_counter()
            try:
                answer = active.answer(question)
            except Exception as exc:
                logger.exception("Evaluation query failed: %s", question)
                samples.append(
                    EvalSample(
                        question=question,
                        ground_truth=str(row.get("ground_truth", "")),
                        answer=f"ERROR: {exc}",
                        unanswerable=bool(row.get("unanswerable", False)),
                        language=str(row.get("language", "en")),
                        grounded=False,
                    )
                )
                continue
            latency = (time.perf_counter() - started) * 1000
            samples.append(
                EvalSample(
                    question=question,
                    ground_truth=str(row.get("ground_truth", "")),
                    answer=answer.answer,
                    contexts=[context.text for context in answer.contexts],
                    sources=[context.citation_key for context in answer.contexts],
                    citations=answer.citation_keys,
                    latency_ms=round(latency, 2),
                    grounded=answer.grounded,
                    unanswerable=bool(row.get("unanswerable", False)),
                    language=str(row.get("language", "en")),
                )
            )
            logger.info("Answered in %.0f ms: %s", latency, question[:70])
        return samples

    # ---------------------------------------------------------- RAGAS mode
    def evaluate_ragas(self, samples: Sequence[EvalSample]) -> tuple[dict[str, float], dict[str, float]]:
        """Compute RAGAS metrics; returns ``(metrics, per_sample_first_metric)``."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise EvaluationError(f"RAGAS is unavailable: {exc}") from exc

        usable = [sample for sample in samples if sample.contexts and not sample.unanswerable]
        if not usable:
            raise EvaluationError("no answerable samples with contexts; cannot run RAGAS")

        dataset = Dataset.from_dict(
            {
                "question": [sample.question for sample in usable],
                "answer": [sample.answer for sample in usable],
                "contexts": [sample.contexts for sample in usable],
                "ground_truth": [sample.ground_truth for sample in usable],
            }
        )
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        ragas_llm = _build_ragas_llm(self.pipeline.llm)
        ragas_embeddings = _build_ragas_embeddings(self.embed_model)
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
                show_progress=False,
            )
        except Exception as exc:
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        scores: dict[str, float] = {}
        for name in METRIC_NAMES:
            value = getattr(result, name, None)
            if value is None and hasattr(result, "scores"):
                value = result.scores.get(name)  # type: ignore[union-attr]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                scores[name] = float(value)

        per_sample: dict[str, float] = {}
        try:
            frame = result.to_pandas()
            for position, row in enumerate(frame.to_dict("records")):
                per_sample[str(position)] = float(row.get("faithfulness", 0.0) or 0.0)
            for position, sample in enumerate(usable):
                if str(position) in per_sample:
                    sample.per_sample_scores["faithfulness"] = per_sample[str(position)]
        except Exception:  # pragma: no cover - pandas optional
            logger.debug("Could not extract per-sample RAGAS scores")
        return scores, per_sample

    # ------------------------------------------------------ heuristic mode
    @staticmethod
    def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
        if not first or not second or len(first) != len(second):
            return 0.0
        dot = sum(a * b for a, b in zip(first, second, strict=False))
        norm_a = math.sqrt(sum(a * a for a in first))
        norm_b = math.sqrt(sum(b * b for b in second))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        unique = list(dict.fromkeys(texts))
        vectors = self.embed_model.get_text_embedding_batch(unique)
        mapping = dict(zip(unique, vectors, strict=True))
        return [mapping[text] for text in texts]

    def evaluate_heuristic(self, samples: Sequence[EvalSample]) -> dict[str, float]:
        """Compute deterministic proxy metrics without an LLM judge."""
        answerable = [sample for sample in samples if sample.contexts and not sample.unanswerable]
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []

        for sample in answerable:
            query_vector = self._embed([sample.question])[0]
            answer_vector = self._embed([sample.answer])[0]
            context_vectors = self._embed(sample.contexts) if sample.contexts else []

            # answer relevancy: question <-> answer similarity
            relevancy_scores.append(self._cosine(query_vector, answer_vector))

            # context precision: rank-weighted relevance of retrieved contexts
            relevance = [self._cosine(query_vector, vector) for vector in context_vectors]
            threshold = max(0.2, (max(relevance) if relevance else 0.0) * 0.7)
            hits = [1.0 if score >= threshold else 0.0 for score in relevance]
            if any(hits):
                precision_sum = 0.0
                found = 0
                for position, hit in enumerate(hits, start=1):
                    if hit:
                        found += 1
                        precision_sum += found / position
                precision_scores.append(precision_sum / found)
            else:
                precision_scores.append(0.0)

            # faithfulness: answer sentences supported by some context sentence
            context_sentences = [sentence for context in sample.contexts for sentence in split_sentences(context)]
            context_sentence_vectors = self._embed(context_sentences) if context_sentences else []
            answer_sentences = [s for s in split_sentences(sample.answer) if tokenize_words(s)]
            if answer_sentences and context_sentence_vectors:
                support = []
                for sentence in answer_sentences:
                    vector = self._embed([sentence])[0]
                    support.append(max(self._cosine(vector, ctx) for ctx in context_sentence_vectors))
                faithfulness_scores.append(sum(support) / len(support))
            else:
                faithfulness_scores.append(0.0)

            # context recall: ground-truth coverage
            truth_sentences = [s for s in split_sentences(sample.ground_truth) if tokenize_words(s)]
            if truth_sentences and context_sentence_vectors:
                coverage = []
                for sentence in truth_sentences:
                    vector = self._embed([sentence])[0]
                    coverage.append(max(self._cosine(vector, ctx) for ctx in context_sentence_vectors))
                recall_scores.append(sum(coverage) / len(coverage))

        def mean(values: Sequence[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        metrics = {
            "faithfulness": mean(faithfulness_scores),
            "answer_relevancy": mean(relevancy_scores),
            "context_precision": mean(precision_scores),
            "context_recall": mean(recall_scores),
        }
        return metrics

    # ------------------------------------------------------------- baseline
    def build_baseline_pipeline(self) -> RAGPipeline:
        """Vector-only retrieval + no reranking (the step-2 baseline)."""
        embed_model = getattr(self.pipeline.multi_retriever.retrievers[0], "index", None)
        embed_model = getattr(embed_model, "embed_model", None)
        vector_retriever = VectorRetriever(
            self.settings,
            embed_model=embed_model,
            weight=1.0,
        )
        return RAGPipeline(
            settings=self.settings,
            db=self.pipeline.db,
            multi_retriever=MultiRetriever([vector_retriever], settings=self.settings),
            reranker=Reranker(backend="none", settings=self.settings, top_n=self.settings.rerank_top_n),
            llm=self.pipeline.llm,
            top_k=self.settings.rerank_top_n,
            rerank_top_n=self.settings.rerank_top_n,
        )

    # ----------------------------------------------------------------- run
    def run(self, with_baseline: bool = True) -> EvaluationReport:
        """Evaluate the advanced pipeline (and optionally the vector baseline)."""
        questions = self.load_testset()
        mode = self.resolve_mode()
        started = time.perf_counter()

        samples = self.collect(questions)
        baseline_samples: list[EvalSample] = []
        baseline_metrics: dict[str, float] = {}
        if with_baseline:
            logger.info("Running the vector-only baseline for comparison")
            baseline_samples = self.collect(questions, pipeline=self.build_baseline_pipeline())

        if mode == "ragas":
            try:
                metrics = self.evaluate_ragas(samples)[0]
                if with_baseline:
                    baseline_metrics = self.evaluate_ragas(baseline_samples)[0]
            except EvaluationError as exc:
                logger.warning("RAGAS metrics failed (%s); falling back to heuristics", exc)
                mode = "heuristic"
                metrics = self.evaluate_heuristic(samples)
                if with_baseline:
                    baseline_metrics = self.evaluate_heuristic(baseline_samples)
        else:
            metrics = self.evaluate_heuristic(samples)
            if with_baseline:
                baseline_metrics = self.evaluate_heuristic(baseline_samples)

        unanswerable = [sample for sample in samples if sample.unanswerable]
        refusal_correct = sum(1 for sample in unanswerable if not sample.grounded)
        extra: dict[str, Any] = {
            "answerable_questions": len(samples) - len(unanswerable),
            "unanswerable_questions": len(unanswerable),
            "refusal_accuracy": round(refusal_correct / len(unanswerable), 3) if unanswerable else None,
            "llm_provider": self.pipeline.llm.provider,
            "reranker": self.pipeline.reranker.description,
            "retrievers": [retriever.name for retriever in self.pipeline.multi_retriever.retrievers],
            "duration_seconds": round(time.perf_counter() - started, 1),
        }
        report = EvaluationReport(
            mode=mode,
            metrics=metrics,
            samples=samples,
            baseline_metrics=baseline_metrics,
            config={
                "testset": str(self.testset_path),
                "questions": len(questions),
                "top_k": self.pipeline.top_k,
                "rerank_top_n": self.pipeline.rerank_top_n,
                "embedding_backend": self.settings.embedding_backend,
                "embedding_model": self.settings.embedding_model,
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.resolved_llm_model,
                "chunk_size_tokens": self.settings.chunk_size_tokens,
                "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
                "sentence_window_size": self.settings.sentence_window_size,
            },
            extra=extra,
        )
        report.output_paths = {key: str(path) for key, path in self.save(report).items()}
        return report

    def save(self, report: EvaluationReport) -> dict[str, Path]:
        """Write JSON + markdown reports under ``output_dir``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = self.output_dir / f"ragas_report_{stamp}.json"
        md_path = self.output_dir / f"ragas_report_{stamp}.md"
        latest_json = self.output_dir / "ragas_report_latest.json"
        latest_md = self.output_dir / "ragas_report_latest.md"
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        markdown = report.to_markdown()
        for path, content in ((json_path, payload), (latest_json, payload), (md_path, markdown), (latest_md, markdown)):
            path.write_text(content, encoding="utf-8")
        logger.info("Evaluation report written to %s", json_path)
        return {"json": json_path, "markdown": md_path, "latest_json": latest_json, "latest_markdown": latest_md}


__all__ = [
    "RAGASEvaluator",
    "EvaluationReport",
    "EvalSample",
    "METRIC_NAMES",
    "DEFAULT_TESTSET",
]
