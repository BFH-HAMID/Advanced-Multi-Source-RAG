"""Knowledge-graph index: entities + relations extracted with an LLM.

Pipeline:
    1. each chunk is sent to the configured LLM with a strict JSON schema and
       asked for ``(subject, relation, object)`` triples;
    2. triples are normalised and inserted into a ``networkx.MultiDiGraph``
       (nodes = entities, edges = relations, both carrying the chunk ids that
       support them);
    3. the graph **and** a chunk store are persisted to
       ``data/indexes/graph/graph.json`` so retrieval works after a restart;
    4. the retriever links query entities, expands ``max_hops`` neighbours and
       returns the best supporting chunks.

When no LLM is configured (or extraction fails) a deterministic multilingual
heuristic extractor is used, so the graph never blocks the pipeline.

Example:
    >>> graph = GraphIndex(settings)
    >>> graph.build(chunks)
    >>> graph.stats()["entities"] >= 1
    True
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from src.config import Settings, get_settings
from src.errors import IndexingError
from src.models import Chunk, RetrievedContext, utc_now_iso
from src.processing.metadata import STOPWORDS
from src.text_utils import normalise_whitespace, split_sentences, tokenize_words

logger = logging.getLogger(__name__)

GRAPH_VERSION = 1

# Latin proper-noun-ish spans: "Acme Corp", "Dhaka", "ISO-27001".
_LATIN_ENTITY_RE = re.compile(r"\b([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")
_NUMBER_ENTITY_RE = re.compile(r"\b(\d{2,4})\b")

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:is|are|was|were|be|being|হলো|হয়|হচ্ছে|হয়েছিল)\b", re.IGNORECASE), "is"),
    (re.compile(r"\b(?:has|have|had|includes?|contains?|আছে|থাকে|অন্তর্ভুক্ত)\b", re.IGNORECASE), "has"),
    (re.compile(r"\b(?:founded|established|প্রতিষ্ঠিত)\b", re.IGNORECASE), "founded"),
    (re.compile(r"\b(?:acquired|bought|কিনেছে|অধিগ্রহণ)\b", re.IGNORECASE), "acquired"),
    (re.compile(r"\b(?:located|based|headquartered|অবস্থিত)\b", re.IGNORECASE), "located_in"),
    (re.compile(r"\b(?:supports?|enables?|সমর্থন)\b", re.IGNORECASE), "supports"),
    (re.compile(r"\b(?:requires?|needs?|প্রয়োজন)\b", re.IGNORECASE), "requires"),
    (re.compile(r"\b(?:provides?|offers?|delivers?|সেবা|প্রদান)\b", re.IGNORECASE), "provides"),
    (re.compile(r"\b(?:uses?|using|ব্যবহার)\b", re.IGNORECASE), "uses"),
    (re.compile(r"\b(?:manages?|owns?|operates?|পরিচালনা)\b", re.IGNORECASE), "manages"),
    (re.compile(r"\b(?:reports?|belongs?|অন্তর্গত)\b", re.IGNORECASE), "belongs_to"),
    (re.compile(r"\b(?:works?|collaborates?|সহযোগিতা)\b", re.IGNORECASE), "works_with"),
)

_ENTITY_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "row",
    "table",
    "columns",
    "page",
    "chapter",
    "section",
    "figure",
    "http",
    "https",
    "www",
    "ok",
    "yes",
    "no",
}


@dataclass(slots=True)
class Triple:
    """A single extracted knowledge-graph triple."""

    subject: str
    relation: str
    object: str
    chunk_id: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class GraphBuildStats:
    """Statistics returned by :meth:`GraphIndex.build`."""

    chunks: int = 0
    triples: int = 0
    entities: int = 0
    relations: int = 0
    llm_used: bool = False
    failures: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "triples": self.triples,
            "entities": self.entities,
            "relations": self.relations,
            "llm_used": self.llm_used,
            "failures": self.failures,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors[:10],
        }


class TripleExtractor(Protocol):
    """Protocol for pluggable triple extractors (LLM or heuristic)."""

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Return ``(subject, relation, object)`` triples for ``text``."""


EXTRACTION_PROMPT = """You are an information-extraction engine building a knowledge graph.
Extract factual (subject, relation, object) triples from the text.

Rules:
- Work in the language of the text (Bangla or English) but keep relations as short English snake_case verbs
  (e.g. is, has, founded, located_in, acquired, provides, requires, supports, manages, belongs_to, works_with).
- Subjects and objects must be short noun phrases (1-6 words), no pronouns, no full sentences.
- Extract at most {max_triples} triples. Skip vague statements.
- If nothing can be extracted, return an empty list.

Return ONLY valid JSON, no markdown fences, in exactly this shape:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Text:
\"\"\"
{text}
\"\"\"
"""


class HeuristicTripleExtractor:
    """Deterministic, dependency-free triple extractor (English + Bangla).

    Used as a fallback when no LLM is configured.  It combines capitalised
    Latin spans, acronyms and frequent Bengali content words with a small set of
    verb patterns to infer relations.
    """

    def __init__(self, max_triples: int = 12, keywords: Sequence[str] | None = None) -> None:
        self.max_triples = max_triples
        self.keywords = [word for word in (keywords or []) if len(word) >= 4]

    def _candidates(self, sentence: str) -> list[str]:
        candidates: list[str] = []
        for match in _LATIN_ENTITY_RE.finditer(sentence):
            phrase = normalise_whitespace(match.group(1)).strip(" .,;:()[]\"'")
            if phrase.lower() in _ENTITY_STOPWORDS:
                continue
            if len(phrase) >= 2:
                candidates.append(phrase)
        for match in _ACRONYM_RE.finditer(sentence):
            acronym = match.group(1)
            if acronym not in _ENTITY_STOPWORDS:
                candidates.append(acronym)
        tokens = tokenize_words(sentence)
        for keyword in self.keywords:
            if keyword in tokens:
                candidates.append(keyword)
        if len(candidates) < 2:
            # Bengali content words (no capitalisation available).
            for token in tokens:
                if len(token) >= 4 and token not in STOPWORDS and not token.isdigit():
                    candidates.append(token)
        seen: set[str] = set()
        unique: list[str] = []
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        # Drop sub-phrases of longer candidates ("acme" when "Acme Bangladesh Ltd"
        # is already an entity in this sentence) to keep the graph clean.
        filtered = [
            candidate
            for candidate in unique
            if not any(
                candidate != other and candidate.lower() in other.lower() for other in unique
            )
        ]
        return (filtered or unique)[:6]

    @staticmethod
    def _relation(sentence: str) -> str:
        for pattern, relation in _RELATION_PATTERNS:
            if pattern.search(sentence):
                return relation
        return "related_to"

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract triples from ``text`` using pattern heuristics."""
        triples: list[tuple[str, str, str]] = []
        for sentence in split_sentences(text):
            if len(sentence) < 12:
                continue
            entities = self._candidates(sentence)
            if len(entities) < 2:
                continue
            relation = self._relation(sentence)
            pairs = [(entities[0], entities[1])]
            if len(entities) >= 3:
                pairs.append((entities[0], entities[2]))
            if len(entities) >= 4:
                pairs.append((entities[2], entities[3]))
            for subject, obj in pairs:
                if subject.lower() == obj.lower():
                    continue
                triples.append((subject, relation, obj))
                if len(triples) >= self.max_triples:
                    return triples
        return triples


def normalise_entity(name: str) -> str:
    """Normalise an entity label for id generation and matching."""
    cleaned = normalise_whitespace(str(name or ""))
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned, flags=re.UNICODE)
    return cleaned.lower()


def normalise_relation(relation: str) -> str:
    """Normalise a relation label to ``snake_case``."""
    cleaned = re.sub(r"[^\w\s\u0980-\u09FF-]", "", str(relation or "").strip().lower())
    cleaned = re.sub(r"[\s-]+", "_", cleaned).strip("_")
    return cleaned or "related_to"


class GraphIndex:
    """Entity/relation graph with persistence and graph-based retrieval."""

    retriever_name = "graph"
    max_label_length = 120

    def __init__(
        self,
        settings: Settings | None = None,
        graph_path: Path | str | None = None,
        llm: Any | None = None,
        max_hops: int | None = None,
        use_llm: bool | None = None,
    ) -> None:
        """
        Args:
            settings: Project settings.
            graph_path: Persistence file (defaults to ``data/indexes/graph/graph.json``).
            llm: Optional LLM client exposing ``complete_json(prompt)``.
            max_hops: Neighbour expansion depth for retrieval.
            use_llm: Force-enable/disable LLM extraction (``None`` = auto).
        """
        self.settings = settings or get_settings()
        self.graph_path = Path(graph_path or self.settings.graph_path)
        self.max_hops = int(max_hops if max_hops is not None else self.settings.graph_max_hops)
        self.use_llm = use_llm
        self._llm = llm
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.chunk_store: dict[str, dict[str, Any]] = {}
        self.created_at: str = utc_now_iso()
        self.updated_at: str = self.created_at
        self.last_build_stats: GraphBuildStats | None = None
        self._loaded = False

    # ------------------------------------------------------------------ LLM
    def _get_llm(self) -> Any | None:
        """Return the LLM client (if usable) for triple extraction."""
        if self.use_llm is False:
            return None
        if self._llm is not None:
            return self._llm
        try:
            from src.generation.llm import get_llm_client

            client = get_llm_client(self.settings)
            if getattr(client, "is_available", True) and client.provider != "extractive":
                self._llm = client
                return client
            logger.info("Graph extraction falls back to heuristics (LLM provider=%s)", client.provider)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("No LLM available for graph extraction (%s); using heuristics", exc)
        return None

    def _extract_with_llm(self, text: str, max_triples: int) -> list[tuple[str, str, str]]:
        """Ask the LLM for triples and parse the JSON response."""
        llm = self._get_llm()
        if llm is None:
            return []
        prompt = EXTRACTION_PROMPT.format(max_triples=max_triples, text=text[:4000])
        payload = llm.complete_json(prompt)
        if not isinstance(payload, dict):
            return []
        raw_triples = payload.get("triples") or payload.get("result") or []
        triples: list[tuple[str, str, str]] = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or item.get("entity1") or "").strip()
            relation = str(item.get("relation") or item.get("relationship") or "").strip()
            obj = str(item.get("object") or item.get("entity2") or "").strip()
            if subject and obj and relation:
                triples.append((subject[: self.max_label_length], relation, obj[: self.max_label_length]))
        return triples[:max_triples]

    # ---------------------------------------------------------------- build
    def _add_triple(self, triple: Triple, chunk: Chunk) -> bool:
        subject_key = normalise_entity(triple.subject)
        object_key = normalise_entity(triple.object)
        if not subject_key or not object_key or subject_key == object_key:
            return False
        if len(subject_key) < self.settings.graph_min_entity_len:
            return False
        if len(object_key) < self.settings.graph_min_entity_len:
            return False

        for key, label in ((subject_key, triple.subject), (object_key, triple.object)):
            if key not in self.graph:
                self.graph.add_node(
                    key,
                    label=label.strip()[: self.max_label_length],
                    entity_type="concept",
                    chunk_ids=[],
                    frequency=0,
                )
            data = self.graph.nodes[key]
            if chunk.chunk_id not in data["chunk_ids"]:
                data["chunk_ids"].append(chunk.chunk_id)
            data["frequency"] = int(data.get("frequency", 0)) + 1

        relation = normalise_relation(triple.relation)
        edge_key = f"{subject_key}|{relation}|{object_key}|{chunk.chunk_id}"
        self.graph.add_edge(
            subject_key,
            object_key,
            key=edge_key,
            relation=relation,
            chunk_id=chunk.chunk_id,
            chunk_ids=[chunk.chunk_id],
            weight=1.0,
            confidence=float(triple.confidence),
        )
        return True

    def build(
        self,
        chunks: Sequence[Chunk],
        persist: bool = True,
        progress: Any | None = None,
    ) -> GraphBuildStats:
        """Extract triples for ``chunks`` and build the graph from scratch.

        Args:
            chunks: Processed chunks.
            persist: Write ``graph.json`` when finished.
            progress: Optional callable ``(done, total)`` for UI progress.

        Returns:
            Build statistics.
        """
        started = time.perf_counter()
        stats = GraphBuildStats()
        self.graph = nx.MultiDiGraph()
        self.chunk_store = {}
        self.created_at = utc_now_iso()

        total = len(chunks)
        if total == 0:
            raise IndexingError("cannot build a graph index without chunks")

        llm = self._get_llm()
        stats.llm_used = llm is not None
        heuristic = HeuristicTripleExtractor()
        max_triples = int(self.settings.graph_max_entities_per_chunk)

        for position, chunk in enumerate(chunks):
            self.chunk_store[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "source_type": str(chunk.source_type),
                "page": chunk.page,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
            raw_triples: list[tuple[str, str, str]] = []
            if llm is not None:
                try:
                    raw_triples = self._extract_with_llm(chunk.text, max_triples)
                except Exception as exc:
                    stats.failures += 1
                    stats.errors.append(f"{chunk.chunk_id}: {exc}")
                    logger.warning("LLM triple extraction failed for %s: %s", chunk.chunk_id, exc)
            if not raw_triples:
                heuristic.keywords = list(chunk.metadata.get("keywords", []) or [])  # type: ignore[arg-type]
                raw_triples = heuristic.extract(chunk.text)

            for subject, relation, obj in raw_triples[:max_triples]:
                if self._add_triple(Triple(subject, relation, obj, chunk.chunk_id), chunk):
                    stats.triples += 1

            stats.chunks += 1
            if progress is not None and (position % 10 == 0 or position == total - 1):
                # A UI callback must never break the build.
                with contextlib.suppress(Exception):
                    progress(position + 1, total)

        stats.entities = self.graph.number_of_nodes()
        stats.relations = self.graph.number_of_edges()
        stats.duration_ms = (time.perf_counter() - started) * 1000
        self.updated_at = utc_now_iso()
        if persist:
            self.persist()
        logger.info(
            "Graph built: %s chunks, %s triples, %s entities, %s relations (llm=%s) in %.0f ms",
            stats.chunks,
            stats.triples,
            stats.entities,
            stats.relations,
            stats.llm_used,
            stats.duration_ms,
        )
        self.last_build_stats = stats
        self._loaded = True
        return stats

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph and chunk store to a plain dictionary."""
        nodes = [
            {
                "id": node_id,
                "label": data.get("label", node_id),
                "entity_type": data.get("entity_type", "concept"),
                "chunk_ids": list(data.get("chunk_ids", [])),
                "frequency": int(data.get("frequency", 0)),
            }
            for node_id, data in self.graph.nodes(data=True)
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "relation": data.get("relation", "related_to"),
                "chunk_id": data.get("chunk_id", ""),
                "chunk_ids": list(data.get("chunk_ids", [])),
                "weight": float(data.get("weight", 1.0)),
                "confidence": float(data.get("confidence", 0.5)),
            }
            for source, target, data in self.graph.edges(data=True)
        ]
        return {
            "version": GRAPH_VERSION,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "settings": {
                "max_hops": self.max_hops,
                "llm_used": bool(self.last_build_stats.llm_used) if self.last_build_stats else False,
            },
            "nodes": nodes,
            "edges": edges,
            "chunks": self.chunk_store,
            "stats": {
                "entities": len(nodes),
                "relations": len(edges),
                "chunks": len(self.chunk_store),
            },
        }

    def persist(self, path: Path | str | None = None) -> Path:
        """Write the graph to ``graph.json`` and return its path."""
        target = Path(path or self.graph_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
            target.write_text(payload, encoding="utf-8")
        except Exception as exc:
            raise IndexingError(f"failed to persist graph: {exc}") from exc
        logger.info("Graph persisted to %s (%s entities)", target, self.graph.number_of_nodes())
        return target

    def exists(self) -> bool:
        """Return ``True`` when a persisted graph file is present."""
        return self.graph_path.exists()

    def load(self) -> GraphIndex:
        """Load a persisted graph (no-op if already in memory)."""
        if self._loaded and self.graph.number_of_nodes() > 0:
            return self
        if not self.graph_path.exists():
            raise IndexingError(
                f"no graph index found at {self.graph_path}; build indexes first"
            )
        try:
            payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IndexingError(f"failed to read graph file: {exc}") from exc

        self.graph = nx.MultiDiGraph()
        self.chunk_store = {
            str(chunk_id): dict(data) for chunk_id, data in (payload.get("chunks") or {}).items()
        }
        for node in payload.get("nodes", []):
            self.graph.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                entity_type=node.get("entity_type", "concept"),
                chunk_ids=list(node.get("chunk_ids", [])),
                frequency=int(node.get("frequency", 0)),
            )
        for edge in payload.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                key=f"{edge['source']}|{edge.get('relation', 'related_to')}|{edge['target']}|{edge.get('chunk_id', '')}",
                relation=edge.get("relation", "related_to"),
                chunk_id=edge.get("chunk_id", ""),
                chunk_ids=list(edge.get("chunk_ids", [])),
                weight=float(edge.get("weight", 1.0)),
                confidence=float(edge.get("confidence", 0.5)),
            )
        self.created_at = str(payload.get("created_at", self.created_at))
        self.updated_at = str(payload.get("updated_at", self.updated_at))
        self._loaded = True
        logger.info(
            "Graph loaded from %s: %s nodes, %s edges, %s chunks",
            self.graph_path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.chunk_store),
        )
        return self

    def clear(self) -> None:
        """Drop the in-memory graph and its persisted file."""
        self.graph = nx.MultiDiGraph()
        self.chunk_store = {}
        self._loaded = False
        if self.graph_path.exists():
            self.graph_path.unlink()

    # ------------------------------------------------------------- retrieval
    def find_entities(self, query: str, limit: int = 12) -> list[tuple[str, float]]:
        """Link query text to graph entities.

        Scoring combines exact substring matches (strongest) with token overlap
        so that multi-word and inflected forms still match.

        Args:
            query: User query.
            limit: Maximum number of entities.

        Returns:
            ``(entity_id, score)`` pairs sorted by descending score.
        """
        if not query or self.graph.number_of_nodes() == 0:
            return []
        query_norm = normalise_entity(query)
        query_tokens = set(tokenize_words(query))
        if not query_tokens:
            return []

        matches: list[tuple[str, float]] = []
        for node_id, data in self.graph.nodes(data=True):
            label = str(data.get("label", node_id))
            label_norm = normalise_entity(label)
            if not label_norm:
                continue
            score = 0.0
            if len(label_norm) >= 3 and re.search(rf"(?<!\w){re.escape(label_norm)}(?!\w)", query_norm):
                score = 1.0 + min(0.4, 0.05 * len(tokenize_words(label_norm)))
            else:
                label_tokens = set(tokenize_words(label))
                if label_tokens:
                    overlap = len(label_tokens & query_tokens)
                    if overlap:
                        score = 0.6 * (overlap / len(label_tokens))
            if score > 0:
                matches.append((node_id, score))
        matches.sort(key=lambda item: (-item[1], item[0]))
        return matches[:limit]

    def expand_chunks(self, entity_ids: Iterable[str], max_hops: int | None = None) -> list[tuple[str, float]]:
        """Collect chunk ids attached to ``entity_ids`` and their neighbours.

        Args:
            entity_ids: Seed entities (from :meth:`find_entities`).
            max_hops: BFS depth (defaults to the configured value).

        Returns:
            ``(chunk_id, score)`` pairs sorted by descending score.
        """
        hops = int(max_hops if max_hops is not None else self.max_hops)
        seeds = list(entity_ids)
        if not seeds:
            return []

        chunk_scores: dict[str, float] = defaultdict(float)
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int, float]] = deque((seed, 0, 1.0) for seed in seeds)

        while queue:
            node_id, depth, weight = queue.popleft()
            if node_id not in self.graph:
                continue
            if node_id in visited and visited[node_id] <= depth:
                continue
            visited[node_id] = depth

            for chunk_id in self.graph.nodes[node_id].get("chunk_ids", []):
                chunk_scores[str(chunk_id)] += weight

            if depth >= hops:
                continue

            # Outgoing relations: strong evidence (subject/object co-occurrence).
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                for chunk_id in edge_data.get("chunk_ids", []):
                    chunk_scores[str(chunk_id)] += weight * 0.6 * float(edge_data.get("weight", 1.0))
                queue.append((str(target), depth + 1, weight * 0.5))

            # Incoming relations: relation still connects both entities.
            for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                for chunk_id in edge_data.get("chunk_ids", []):
                    chunk_scores[str(chunk_id)] += weight * 0.5 * float(edge_data.get("weight", 1.0))
                queue.append((str(source), depth + 1, weight * 0.45))

        if not chunk_scores:
            return []
        max_score = max(chunk_scores.values()) or 1.0
        ranked = [(chunk_id, score / max_score) for chunk_id, score in chunk_scores.items()]
        ranked.sort(key=lambda item: -item[1])
        return ranked

    def search(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Retrieve chunks by graph traversal from query entities."""
        if self.graph.number_of_nodes() == 0:
            self.load()
        entities = self.find_entities(query)
        if not entities:
            logger.debug("Graph retriever found no entities for %r", query[:60])
            return []
        ranked_chunks = self.expand_chunks([entity_id for entity_id, _ in entities])
        contexts: list[RetrievedContext] = []
        for chunk_id, score in ranked_chunks[: max(1, top_k)]:
            payload = self.chunk_store.get(chunk_id)
            if not payload:
                continue
            contexts.append(
                RetrievedContext(
                    node_id=chunk_id,
                    text=str(payload.get("text", "")),
                    score=float(score),
                    source=str(payload.get("source", "")),
                    page=int(payload["page"]) if payload.get("page") is not None else None,
                    source_type=str(payload.get("source_type", "text")),
                    retriever=self.retriever_name,
                    retrievers=[self.retriever_name],
                    metadata={
                        "chunk_id": chunk_id,
                        "matched_entities": [self.graph.nodes[e]["label"] for e, _ in entities[:5]],
                        "graph_score": float(score),
                    },
                )
            )
        for rank, context in enumerate(contexts):
            context.rank = rank
        logger.debug("Graph search returned %s contexts via %s entities", len(contexts), len(entities))
        return contexts

    # ------------------------------------------------------------------- info
    def stats(self) -> dict[str, Any]:
        """Return graph size statistics (loads the graph if needed)."""
        if self.graph.number_of_nodes() == 0 and self.exists() and not self._loaded:
            try:
                self.load()
            except IndexingError:  # pragma: no cover - corrupted graph file
                logger.warning("Could not load graph statistics from %s", self.graph_path)
        relations: dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relations[str(data.get("relation", "related_to"))] += 1
        top_entities = sorted(
            (
                (data.get("label", node_id), int(data.get("frequency", 0)))
                for node_id, data in self.graph.nodes(data=True)
            ),
            key=lambda item: -item[1],
        )[:10]
        return {
            "entities": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "chunks": len(self.chunk_store),
            "relation_types": dict(sorted(relations.items(), key=lambda item: -item[1])[:10]),
            "top_entities": top_entities,
            "updated_at": self.updated_at,
            "path": str(self.graph_path),
        }

    def export_triples(self, limit: int = 50) -> list[dict[str, str]]:
        """Return readable triples for the UI (``subject –relation→ object``)."""
        triples: list[dict[str, str]] = []
        for source, target, data in self.graph.edges(data=True):
            triples.append(
                {
                    "subject": str(self.graph.nodes[source].get("label", source)),
                    "relation": str(data.get("relation", "related_to")),
                    "object": str(self.graph.nodes[target].get("label", target)),
                    "chunk_id": str(data.get("chunk_id", "")),
                }
            )
            if len(triples) >= limit:
                break
        return triples


__all__ = [
    "GraphIndex",
    "Triple",
    "GraphBuildStats",
    "TripleExtractor",
    "HeuristicTripleExtractor",
    "normalise_entity",
    "normalise_relation",
    "EXTRACTION_PROMPT",
]
