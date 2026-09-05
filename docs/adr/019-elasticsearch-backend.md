# ADR 019: Elasticsearch 9.x Backend Architecture & Hybrid Retrieval

## Status
Accepted

## Context
MemoryGraph captures agent experiences as an interconnected knowledge graph with 35+ typed relationships (`SOLVES`, `CAUSES`, `DEPENDS_ON`, `PREVENTS`), bi-temporal validity tracking, and memory importance/confidence weighting.

Existing backends comprise:
1. Native Cypher graph databases (FalkorDB, FalkorDBLite, Neo4j, Memgraph).
2. Embedded relational fallback (SQLite) with simulated BFS traversal in TypeScript.
3. MemoryGraph Cloud API.

While graph databases excel at multi-hop relational queries, standard text retrieval in FalkorDB and SQLite is limited to exact substring matching (`CONTAINS` / `LIKE '%query%'`). In AI coding workflows, agents often express search queries with different vocabulary than the stored memory (e.g. searching "docker daemon socket non-root" when the memory contains "EACCES: permission denied, open /var/run/docker.sock").

Elasticsearch 9.x (and Elastic Cloud Serverless) provides a unified platform combining:
- BM25 full-text indexing with analyzers and fuzziness.
- Dense vector similarity and `semantic_text` automated embeddings via Lucene 10.
- ES|QL (Elasticsearch Query Language) with `LOOKUP JOIN` for single-query relationship traversal.
- Hybrid keyword + vector retrieval.

## Decision

We introduce `ElasticsearchBackend` in `ts/src/backends/elasticsearch.ts` implementing the `GraphBackend` interface.

### 1. Architectural Strategy
- **Zero Host Pollution & Zero External SDKs**: The client communicates over HTTP/HTTPS using the runtime's global `fetch` API. No `@elastic/elasticsearch` client package or Python runtime is required, keeping the compiled Bun binary and Docker container lightweight.
- **Elastic Cloud Serverless & Local 9.5+ Support**: Accepts standard endpoint URL (`MEMORY_ELASTICSEARCH_URL`) and API key (`MEMORY_ELASTICSEARCH_API_KEY`) or Basic Auth.
- **Environment Auto-Detection (`MEMORY_ELASTICSEARCH_SEMANTIC_SEARCH`)**:
  - Defaults to `true` when connecting to Elastic Cloud (`*.elastic.cloud`), where the Elastic Inference Service is turnkey.
  - Defaults to `false` when connecting to local/self-hosted Elasticsearch, running high-speed BM25 search out of the box without requiring local ML model setup.

### 2. Multi-Field Mapping Architecture
To achieve true hybrid search, `content` and `summary` use a **multi-field mapping pattern**:
```json
{
  "properties": {
    "title": { "type": "text" },
    "content": {
      "type": "text",
      "fields": {
        "semantic": { "type": "semantic_text" }
      }
    },
    "summary": {
      "type": "text",
      "fields": {
        "semantic": { "type": "semantic_text" }
      }
    },
    "tags": { "type": "keyword" },
    "importance": { "type": "float" },
    "confidence": { "type": "float" }
  }
}
```
- **`content` (root)**: Standard `text` field providing full Lucene 10 BM25 tokenization, inverted indexing, and stemming for exact keyword precision.
- **`content.semantic` (subfield)**: `semantic_text` field providing automated vector embedding inference for conceptual similarity.

### 3. Retrieval Tuning & Scoring Strategy

#### A. Vector Score Balancing (`boost: 10.0`)
Raw BM25 lexical scores are unbounded (typically 5.0 to 25.0+ on strong matches), whereas cosine vector similarity scores are bounded between 0.0 and 1.0. To bring semantic similarity onto the same numerical scale and ensure conceptual intent is given equal weight alongside keyword matches, `content.semantic` is boosted by `10.0` and `summary.semantic` by `5.0`.

#### B. Multi-Word Quorum (`minimum_should_match: "2<75%"`)
For natural language queries containing multiple words, adding `minimum_should_match: "2<75%"` ensures queries with 3 or more terms require at least 2 matching terms, prioritizing documents with comprehensive topical coverage.

#### C. Dash & Punctuation Normalization
Typographic en-dashes (`–`) and em-dashes (`—`) are normalized to standard ASCII hyphens (`-`) before query execution to ensure seamless matching across Wikipedia titles and technical compound identifiers.

### 4. Graph Traversal & Temporal Versioning
- **Deterministic BFS**: For multi-hop graph expansion (`getRelatedMemories`), the backend queries `${prefix}_relationships` by node IDs up to `maxDepth`.
- **Bi-Temporal Versioning**: Every call to `updateMemory` archives an immutable snapshot to `${prefix}_versions`, supporting historical state queries (`getMemoryStateAt`, `getMemoryVersions`).

## Consequences

### Positive
- True hybrid search combining exact technical symbol precision with conceptual intent.
- Resilient to typos, vocabulary mismatch, and complex technical abstractions.
- Turnkey local execution without requiring an Elastic Cloud account or ML node setup.
- Zero host pollution and zero external client SDK dependencies.

### Negative / Trade-offs
- Ingestion in Elastic Cloud with server-side vector inference is slower (~95–115 docs/sec) than local SQLite or FalkorDB (~1,000 docs/sec).
- Round-trip query latency to Elastic Cloud (~75–85ms) is higher than in-process embedded storage (~2–15ms).
