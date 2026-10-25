# Architecture & design decisions

This document explains *why* the system is built the way it is. For the
walkthrough and run instructions see the [README](../README.md).

---

## 1. Component map

```
                    ┌──────────────────────── app/streamlit_app.py ───────────────────────┐
                    │ uploads · URLs · chat · sources · index status · RAGAS runner       │
                    └───────────────┬────────────────────────────────────────────────────┘
                                    │ src/pipeline.RAGPipeline
   ┌────────────────────────────────┼────────────────────────────────────────────────────┐
   │                                ▼                                                    │
   │  ingestion/ ──► storage/document_db ──► processing/ ──► indexing/ ──► retrieval/    │
   │  loaders          documents              cleaner          vector       vector       │
   │  pipeline         chunks                 chunker          window       window       │
   │                   ingestion_log          metadata         graph        graph        │
   │                   app_meta                                            multi        │
   │                                     ▼                                              │
   │                                  rerank/ ──► generation/ ──► evaluation/            │
   │                                  reranker     query_builder  ragas_eval             │
   │                                               llm · formatter                       │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Unified Document DB (SQLite)

```sql
documents(doc_id PK, source, source_type, page, title, text, char_count,
          content_hash, created_at, metadata JSON)
chunks(chunk_id PK, doc_id FK, source, source_type, page, title, chunk_index,
       text, token_count, created_at, metadata JSON)
ingestion_log(id PK, source, source_type, status, message, doc_count, chunk_count, created_at)
app_meta(key PK, value JSON)
```

* **Why SQLite?** Zero-ops, single file, transactional, good enough for
  departmental corpora (≤ millions of chunks with indexes). The repository layer
  is isolated in one class, so swapping in PostgreSQL means re-implementing
  `DocumentDB` only.
* **WAL + one connection per call** keeps it safe under Streamlit reruns and the
  retriever thread pool (SQLite objects are not shareable across threads).
* **Raw text is never destroyed** by re-chunking, so changing `CHUNK_SIZE_TOKENS`
  costs only a `--reprocess`, not a re-upload.
* `doc_id` and `chunk_id` are content-derived hashes → re-ingesting the same file
  is idempotent, and every index can be rebuilt deterministically.

## 3. Indexing

| Index | Storage | Build cost | Rebuild trigger |
| --- | --- | --- | --- |
| Vector | `data/indexes/vector/` (FAISS binary + LlamaIndex docstore) | embeddings | embedding model/dimension change |
| Sentence window | `data/indexes/sentence_window/` (FAISS over sentence nodes) | embeddings (more nodes) | `SENTENCE_WINDOW_SIZE` change |
| Graph | `data/indexes/graph/graph.json` (nodes, edges, chunk store) | 1 LLM call per chunk | document changes |

* FAISS uses `IndexFlatIP` over L2-normalised vectors → cosine similarity with
  exact search. For >1M chunks, switch to `IndexIVFFlat` or HNSW; the wrapper in
  `vector_index.py` is the only place that changes.
* `meta.json` records the embedding dimension; loading an index built with a
  different model fails loudly instead of returning garbage.
* The **sentence-window** parser uses the project's own Unicode sentence splitter,
  because LlamaIndex's default NLTK splitter does not understand the Bengali
  danda (`।`) — Bangla windows would otherwise degrade to one giant sentence.
* The **graph** is a `networkx.MultiDiGraph`: nodes are entities (normalised to
  lowercase ids, display label kept), edges are relations, and both carry the
  chunk ids that support them, which is what the retriever turns back into text.
  Triple extraction asks the LLM for strict JSON; a deterministic multilingual
  extractor (capitalised spans, acronyms, Bengali content words + verb patterns)
  is used when no LLM is configured.
* The **manifest** (`data/indexes/manifest.json`) stores the chunk fingerprint;
  the UI compares it with the DB to flag stale indexes.

## 4. Retrieval & fusion

```
query ──┬─► vector.search(top_k)        ─┐
        ├─► window.search(top_k)        ─┼─► dedupe by node_id / normalised text
        └─► graph.search(top_k)         ─┘        │
                                                   ▼
                                 RRF: Σ wᵣ / (60 + rankᵣ)
                                                   │
                                                   ▼
                            cross-encoder rerank ─► top 5 ─► prompt
```

* **Parallelism**: `ThreadPoolExecutor` with per-retriever timeouts; a failing
  retriever is reported in `errors` and the query still succeeds.
* **RRF over score blending**: the retrievers produce incomparable scores
  (cosine vs. graph evidence vs. window similarity); rank fusion is scale-free
  and needs no score calibration.
* **Deduplication happens twice**: by id/text during fusion (cheap, exact) and by
  token-set Jaccard (≥ 0.8, same source+page) during reranking, which collapses
  overlapping sentence windows into one passage.
* **Relevance gate**: if the best rerank score is below the backend threshold,
  the pipeline refuses instead of feeding unrelated text to the LLM. This is what
  keeps "What is the capital of France?" honest.

## 5. Prompt & answer contract

* Context blocks are rendered as `[2] [handbook.pdf, p.4]` so the model can copy
  the exact label into the answer; the prompt forbids outside knowledge and
  prescribes the refusal sentence verbatim.
* The context section is token-budgeted (`MAX_CONTEXT_TOKENS`, default 3000). The
  last passage is truncated to fit rather than dropped, so the budget is never
  exceeded.
* Citations are parsed back out of the answer, resolved against the contexts, and
  de-duplicated. If the model forgets to cite, the top passages are cited so the
  UI always answers "where did this come from?".
* `RAGAnswer` carries timings, per-retriever counts, provider/model and fallback
  warnings — every answer is auditable.

## 6. Failure modes and fallbacks

| Failure | Behaviour |
| --- | --- |
| Missing/corrupt PDF | `LoaderError` for that file only; the batch continues and the error is logged in `ingestion_log` |
| Scanned PDF (no text) | explicit error telling the operator OCR is required |
| Embedding model cannot download | falls back to hashing embeddings (logged, visible in UI) |
| Cross-encoder unavailable | lexical IDF reranker |
| No LLM credentials / API down | extractive answer built from the retrieved sentences, still cited |
| One index missing | that retriever is disabled and reported; the others answer |
| Corrupt index / dimension mismatch | loud `IndexingError` with the rebuild command |
| Empty retrieval / low relevance | refusal in the question's language, no invented citation |

## 7. Bilingual (Bangla) considerations

* **Embeddings**: bge-m3 covers Bangla and English in one vector space; no
  separate model or translation step is needed.
* **Tokenisation**: `src/text_utils.py` counts tokens with a Unicode-aware
  estimator (Bengali conjuncts are multiple code points), so 600-token chunks are
  genuinely ~600 tokens rather than silently oversized.
* **Sentence splitting**: handles `।`, `॥`, `?`, `!`, `…` and abbreviation edge
  cases; used by the chunker and the sentence-window parser.
* **Metadata**: language tag (`bn` / `en` / `mixed`), so retrieval can be filtered
  or boosted per language.
* **Prompt**: the system prompt instructs the model to answer in the language of
  the question; refusal messages exist in both languages.
* **Graph extraction**: the heuristic fallback includes Bengali verbs
  (`হলো`, `আছে`, `করে`, `অবস্থিত`, …) so the graph is useful without an LLM.

## 8. Testing strategy

The suite runs without network, models or API keys: hashing embeddings, lexical
reranker and the extractive LLM are first-class backends, not test hacks. This
keeps CI deterministic and fast (~5 s) while still exercising every stage —
including a full ingest → index → ask → evaluate path.
