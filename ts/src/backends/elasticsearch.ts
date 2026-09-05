/**
 * Elasticsearch 9.x / Elastic Cloud Serverless backend for MemoryGraph.
 *
 * Implements GraphBackend using native runtime `fetch` (zero additional npm dependencies).
 * Supports:
 * - Hybrid BM25 full-text + dense vector/semantic_text search
 * - Server-side automated inference via `semantic_text`
 * - Graph BFS traversal and bi-temporal relationship tracking
 * - Bi-temporal memory state versioning (H7 / M5)
 */

import { randomUUID } from "node:crypto";
import { Config } from "../config.ts";
import type {
  Memory,
  Relationship,
  RelationshipProperties,
  SearchQuery,
  MemoryType,
} from "../models.ts";
import {
  createMemory,
  createRelationshipProperties,
  isRelationshipType,
} from "../models.ts";
import type {
  GraphBackend,
  HealthCheckResult,
} from "./base.ts";
import {
  DatabaseConnectionError,
  MemoryNotFoundError,
  RelationshipError,
  ValidationError,
} from "../errors.ts";

export interface ElasticsearchConfig {
  url?: string;
  apiKey?: string;
  username?: string;
  password?: string;
  indexPrefix?: string;
  timeout?: number;
  semanticSearch?: boolean;
}

interface ESRelationDoc {
  id: string;
  from_id: string;
  to_id: string;
  rel_type: string;
  strength: number;
  confidence: number;
  context?: string | null;
  evidence_count: number;
  valid_from: string;
  valid_until?: string | null;
  recorded_at: string;
  invalidated_by?: string | null;
  properties?: string | Record<string, unknown>;
}

function toISO(d: string | Date | undefined | null): string | null {
  if (!d) return null;
  return d instanceof Date ? d.toISOString() : new Date(d).toISOString();
}

export class ElasticsearchBackend implements GraphBackend {
  readonly url: string;
  readonly apiKey?: string;
  readonly username?: string;
  readonly password?: string;
  readonly prefix: string;
  readonly timeout: number;

  readonly memoriesIndex: string;
  readonly relationshipsIndex: string;
  readonly versionsIndex: string;

  private _connected = false;
  private _hasSemanticSupport = false;

  constructor(options?: ElasticsearchConfig) {
    this.url = (options?.url ?? Config.ELASTICSEARCH_URL).replace(/\/+$/, "");
    this.apiKey = options?.apiKey ?? Config.ELASTICSEARCH_API_KEY;
    this.username = options?.username ?? Config.ELASTICSEARCH_USERNAME;
    this.password = options?.password ?? Config.ELASTICSEARCH_PASSWORD;
    this.prefix = options?.indexPrefix ?? Config.ELASTICSEARCH_INDEX_PREFIX;
    this.timeout = options?.timeout ?? Config.ELASTICSEARCH_TIMEOUT;

    this.memoriesIndex = `${this.prefix}_memories`;
    this.relationshipsIndex = `${this.prefix}_relationships`;
    this.versionsIndex = `${this.prefix}_versions`;

    this._hasSemanticSupport = options?.semanticSearch ?? Config.ELASTICSEARCH_SEMANTIC_SEARCH;
  }

  backendName(): string {
    return "Elasticsearch";
  }

  supportsFulltextSearch(): boolean {
    return true;
  }

  supportsTransactions(): boolean {
    return false;
  }

  isCypherCapable(): boolean {
    return false;
  }

  async executeQuery(
    _query: string,
    _parameters?: Record<string, unknown>,
    _write?: boolean
  ): Promise<Record<string, unknown>[]> {
    throw new DatabaseConnectionError(
      "Elasticsearch backend does not support raw Cypher queries. Use storeMemory(), searchMemories(), etc."
    );
  }

  // ---------------------------------------------------------------------------
  // HTTP Client
  // ---------------------------------------------------------------------------

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "memorygraph-es/1.0",
    };

    if (this.apiKey) {
      headers["Authorization"] = `ApiKey ${this.apiKey}`;
    } else if (this.username && this.password) {
      const creds = Buffer.from(`${this.username}:${this.password}`).toString("base64");
      headers["Authorization"] = `Basic ${creds}`;
    }

    return headers;
  }

  private async request<T = Record<string, unknown>>(
    method: string,
    path: string,
    body?: unknown,
    params?: Record<string, string>
  ): Promise<{ status: number; ok: boolean; data: T }> {
    let url = `${this.url}${path.startsWith("/") ? path : `/${path}`}`;
    if (params) {
      const q = new URLSearchParams(params);
      url += `?${q.toString()}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.getHeaders(),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      let data: T;
      const text = await response.text();
      try {
        data = text ? JSON.parse(text) : ({} as T);
      } catch {
        data = text as unknown as T;
      }

      return {
        status: response.status,
        ok: response.ok,
        data,
      };
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new DatabaseConnectionError(`Elasticsearch request timed out after ${this.timeout}ms`);
      }
      throw new DatabaseConnectionError(`Elasticsearch request failed: ${err.message ?? err}`);
    }
  }

  // ---------------------------------------------------------------------------
  // Lifecycle & Health
  // ---------------------------------------------------------------------------

  async connect(): Promise<boolean> {
    try {
      const res = await this.request<any>("GET", "/");
      if (!res.ok) {
        throw new DatabaseConnectionError(
          `Failed to connect to Elasticsearch at ${this.url}: HTTP ${res.status}`
        );
      }
      this._connected = true;
      const version = res.data?.version?.number ?? "unknown";
      const flavor = res.data?.version?.build_flavor ?? "standard";
      console.log(`Successfully connected to Elasticsearch (${flavor} ${version}) at ${this.url}`);
      return true;
    } catch (err: any) {
      this._connected = false;
      throw new DatabaseConnectionError(`Cannot connect to Elasticsearch: ${err.message ?? err}`);
    }
  }

  async disconnect(): Promise<void> {
    this._connected = false;
    console.log("Elasticsearch connection closed");
  }

  async healthCheck(): Promise<HealthCheckResult> {
    try {
      const res = await this.request<any>("GET", "/_cluster/health");
      if (res.ok) {
        return {
          connected: true,
          backend_type: "elasticsearch",
          status: res.data.status,
          cluster_name: res.data.cluster_name,
        };
      }
      // In serverless, cluster health may be restricted; fall back to root info
      const rootRes = await this.request<any>("GET", "/");
      return {
        connected: rootRes.ok,
        backend_type: "elasticsearch",
        version: rootRes.data?.version?.number,
      };
    } catch (err: any) {
      return {
        connected: false,
        backend_type: "elasticsearch",
        error: err.message ?? String(err),
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Schema Initialization
  // ---------------------------------------------------------------------------

  async initializeSchema(): Promise<void> {
    // 1. Initialize memories index
    const memExists = await this.request("HEAD", `/${this.memoriesIndex}`);
    if (memExists.status === 404) {
      const contentMapping = this._hasSemanticSupport
        ? {
            type: "text",
            fields: {
              semantic: { type: "semantic_text" },
            },
          }
        : { type: "text" };

      const mapping = {
        mappings: {
          properties: {
            id: { type: "keyword" },
            type: { type: "keyword" },
            title: { type: "text" },
            content: contentMapping,
            summary: contentMapping,
            tags: { type: "keyword" },
            importance: { type: "float" },
            confidence: { type: "float" },
            effectiveness: { type: "float" },
            usage_count: { type: "integer" },
            created_at: { type: "date" },
            updated_at: { type: "date" },
            last_accessed: { type: "date" },
            valid_from: { type: "date" },
            valid_until: { type: "date" },
            recorded_at: { type: "date" },
            invalidated_by: { type: "keyword" },
            context_project_path: { type: "keyword" },
            context_branch: { type: "keyword" },
            context_language: { type: "keyword" },
            context_framework: { type: "keyword" },
            context_summary: { type: "text" },
            metadata: { type: "object", enabled: true },
          },
        },
      };

      const createRes = await this.request("PUT", `/${this.memoriesIndex}`, mapping);
      if (!createRes.ok) {
        throw new DatabaseConnectionError(
          `Failed to create memories index: ${JSON.stringify(createRes.data)}`
        );
      }
    } else {
      // Index already exists: inspect mapping to detect if semantic_text is active
      try {
        const mapRes = await this.request<any>("GET", `/${this.memoriesIndex}/_mapping`);
        const firstIndex = Object.keys(mapRes.data ?? {})[0];
        const props = mapRes.data?.[firstIndex]?.mappings?.properties;
        this._hasSemanticSupport = Boolean(
          props?.content?.type === "semantic_text" ||
          props?.content?.fields?.semantic?.type === "semantic_text"
        );
      } catch {
        // preserve current setting
      }
    }

    // 2. Initialize relationships index
    const relExists = await this.request("HEAD", `/${this.relationshipsIndex}`);
    if (relExists.status === 404) {
      const relMapping = {
        mappings: {
          properties: {
            id: { type: "keyword" },
            from_id: { type: "keyword" },
            to_id: { type: "keyword" },
            rel_type: { type: "keyword" },
            strength: { type: "float" },
            confidence: { type: "float" },
            context: { type: "text" },
            evidence_count: { type: "integer" },
            valid_from: { type: "date" },
            valid_until: { type: "date" },
            recorded_at: { type: "date" },
            invalidated_by: { type: "keyword" },
            properties: { type: "text" },
          },
        },
      };
      const res = await this.request("PUT", `/${this.relationshipsIndex}`, relMapping);
      if (!res.ok) {
        throw new DatabaseConnectionError(
          `Failed to create relationships index: ${JSON.stringify(res.data)}`
        );
      }
    }

    // 3. Initialize versions index (for bi-temporal versioning H7 / M5)
    const verExists = await this.request("HEAD", `/${this.versionsIndex}`);
    if (verExists.status === 404) {
      const verMapping = {
        mappings: {
          properties: {
            version_id: { type: "keyword" },
            memory_id: { type: "keyword" },
            snapshot_at: { type: "date" },
            recorded_at: { type: "date" },
            payload: { type: "text" },
          },
        },
      };
      await this.request("PUT", `/${this.versionsIndex}`, verMapping);
    }
  }

  // ---------------------------------------------------------------------------
  // Memory CRUD
  // ---------------------------------------------------------------------------

  async storeMemory(memory: Memory): Promise<string> {
    if (!memory.id) {
      memory.id = randomUUID();
    }
    const createdAtStr = toISO(memory.created_at) ?? new Date().toISOString();
    const doc = {
      id: memory.id,
      type: memory.type,
      title: memory.title,
      content: memory.content,
      summary: memory.summary ?? null,
      tags: memory.tags ?? [],
      importance: memory.importance,
      confidence: memory.confidence,
      effectiveness: memory.effectiveness,
      usage_count: memory.usage_count,
      created_at: createdAtStr,
      updated_at: toISO(memory.updated_at) ?? createdAtStr,
      last_accessed: toISO(memory.last_accessed),
      valid_from: toISO(memory.valid_from) ?? createdAtStr,
      valid_until: toISO(memory.valid_until),
      recorded_at: toISO(memory.recorded_at) ?? createdAtStr,
      invalidated_by: memory.invalidated_by ?? null,
      context_project_path: memory.context_project_path ?? null,
      context_branch: memory.context_branch ?? null,
      context_language: memory.context_language ?? null,
      context_framework: memory.context_framework ?? null,
      context_summary: memory.context_summary ?? null,
      metadata: memory.metadata ?? {},
    };

    const res = await this.request(
      "PUT",
      `/${this.memoriesIndex}/_doc/${encodeURIComponent(memory.id!)}`,
      doc,
      { refresh: "wait_for" }
    );

    if (!res.ok) {
      throw new DatabaseConnectionError(`Failed to store memory: ${JSON.stringify(res.data)}`);
    }

    return memory.id!;
  }

  async bulkStoreMemories(memories: Memory[]): Promise<string[]> {
    if (memories.length === 0) return [];
    const lines: string[] = [];
    const ids: string[] = [];

    for (const memory of memories) {
      if (!memory.id) memory.id = randomUUID();
      ids.push(memory.id);
      const createdAtStr = toISO(memory.created_at) ?? new Date().toISOString();

      lines.push(JSON.stringify({ index: { _index: this.memoriesIndex, _id: memory.id } }));
      lines.push(
        JSON.stringify({
          id: memory.id,
          type: memory.type,
          title: memory.title,
          content: memory.content,
          summary: memory.summary ?? null,
          tags: memory.tags ?? [],
          importance: memory.importance,
          confidence: memory.confidence,
          effectiveness: memory.effectiveness,
          usage_count: memory.usage_count,
          created_at: createdAtStr,
          updated_at: toISO(memory.updated_at) ?? createdAtStr,
          last_accessed: toISO(memory.last_accessed),
          valid_from: toISO(memory.valid_from) ?? createdAtStr,
          valid_until: toISO(memory.valid_until),
          recorded_at: toISO(memory.recorded_at) ?? createdAtStr,
          invalidated_by: memory.invalidated_by ?? null,
          context_project_path: memory.context_project_path ?? null,
          context_branch: memory.context_branch ?? null,
          context_language: memory.context_language ?? null,
          context_framework: memory.context_framework ?? null,
          context_summary: memory.context_summary ?? null,
          metadata: memory.metadata ?? {},
        })
      );
    }

    const ndjson = lines.join("\n") + "\n";
    const url = `${this.url}/_bulk?refresh=true`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          ...this.getHeaders(),
          "Content-Type": "application/x-ndjson",
        },
        body: ndjson,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new DatabaseConnectionError(`Bulk insert failed: HTTP ${response.status}`);
      }
      return ids;
    } catch (err: any) {
      clearTimeout(timeoutId);
      throw new DatabaseConnectionError(`Bulk insert failed: ${err.message ?? err}`);
    }
  }

  async bulkCreateRelationships(
    relationships: { from: string; to: string; type: string; properties?: RelationshipProperties }[]
  ): Promise<string[]> {
    if (relationships.length === 0) return [];
    const lines: string[] = [];
    const ids: string[] = [];

    for (const rel of relationships) {
      const relId = randomUUID();
      ids.push(relId);
      const props = createRelationshipProperties(rel.properties ?? {});

      lines.push(JSON.stringify({ index: { _index: this.relationshipsIndex, _id: relId } }));
      lines.push(
        JSON.stringify({
          id: relId,
          from_id: rel.from,
          to_id: rel.to,
          rel_type: rel.type,
          strength: props.strength,
          confidence: props.confidence,
          context: props.context ?? null,
          evidence_count: props.evidence_count,
          valid_from: props.valid_from,
          valid_until: props.valid_until ?? null,
          recorded_at: props.recorded_at,
          invalidated_by: props.invalidated_by ?? null,
          properties: JSON.stringify(props),
        })
      );
    }

    const ndjson = lines.join("\n") + "\n";
    const url = `${this.url}/_bulk?refresh=true`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          ...this.getHeaders(),
          "Content-Type": "application/x-ndjson",
        },
        body: ndjson,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new DatabaseConnectionError(`Bulk relationships insert failed: HTTP ${response.status}`);
      }
      return ids;
    } catch (err: any) {
      clearTimeout(timeoutId);
      throw new DatabaseConnectionError(`Bulk relationships insert failed: ${err.message ?? err}`);
    }
  }

  async getMemory(memoryId: string, includeRelationships = true): Promise<Memory | null> {
    const res = await this.request<any>(
      "GET",
      `/${this.memoriesIndex}/_doc/${encodeURIComponent(memoryId)}`
    );

    if (res.status === 404 || !res.ok || !res.data?._source) {
      return null;
    }

    const src = res.data._source;
    const memory = this.mapSourceToMemory(src);

    if (includeRelationships) {
      const relQuery = {
        query: {
          bool: {
            should: [
              { term: { from_id: memoryId } },
              { term: { to_id: memoryId } },
            ],
          },
        },
        size: 1000,
      };

      const relRes = await this.request<any>(
        "POST",
        `/${this.relationshipsIndex}/_search`,
        relQuery
      );

      if (relRes.ok && relRes.data?.hits?.hits) {
        const rels: Record<string, string[]> = {};
        for (const hit of relRes.data.hits.hits) {
          const r = hit._source;
          const otherId = r.from_id === memoryId ? r.to_id : r.from_id;
          if (!rels[r.rel_type]) rels[r.rel_type] = [];
          if (!rels[r.rel_type].includes(otherId)) {
            rels[r.rel_type].push(otherId);
          }
        }
        memory.relationships = rels;
      }
    }

    return memory;
  }

  async updateMemory(memory: Memory): Promise<boolean> {
    const memoryId = memory.id!;
    // Snapshot old version into versions index (VAL-LOCAL-017..019)
    const existing = await this.getMemory(memoryId, false);
    if (!existing) {
      throw new MemoryNotFoundError(`Memory not found: ${memoryId}`);
    }

    const versionDoc = {
      version_id: randomUUID(),
      memory_id: existing.id!,
      snapshot_at: new Date().toISOString(),
      recorded_at: toISO(existing.updated_at) ?? new Date().toISOString(),
      payload: JSON.stringify(existing),
    };
    await this.request("POST", `/${this.versionsIndex}/_doc`, versionDoc);

    // Save updated memory
    const updated = {
      ...memory,
      updated_at: new Date(),
    };
    await this.storeMemory(updated);
    return true;
  }

  async deleteMemory(memoryId: string): Promise<boolean> {
    const res = await this.request(
      "DELETE",
      `/${this.memoriesIndex}/_doc/${encodeURIComponent(memoryId)}`,
      undefined,
      { refresh: "wait_for" }
    );

    if (res.status === 404) return false;

    // Delete connected relationships
    const deleteRelsQuery = {
      query: {
        bool: {
          should: [
            { term: { from_id: memoryId } },
            { term: { to_id: memoryId } },
          ],
        },
      },
    };
    await this.request(
      "POST",
      `/${this.relationshipsIndex}/_delete_by_query`,
      deleteRelsQuery,
      { refresh: "true" }
    );

    return true;
  }

  // ---------------------------------------------------------------------------
  // Relationships
  // ---------------------------------------------------------------------------

  async createRelationship(
    fromMemoryId: string,
    toMemoryId: string,
    relationshipType: string,
    properties?: RelationshipProperties
  ): Promise<string> {
    if (!isRelationshipType(relationshipType)) {
      throw new ValidationError(`Invalid relationship type: ${relationshipType}`);
    }

    const fromMem = await this.getMemory(fromMemoryId, false);
    if (!fromMem) throw new MemoryNotFoundError(`Source memory not found: ${fromMemoryId}`);

    const toMem = await this.getMemory(toMemoryId, false);
    if (!toMem) throw new MemoryNotFoundError(`Target memory not found: ${toMemoryId}`);

    const relId = randomUUID();
    const props = createRelationshipProperties(properties ?? {});

    const doc: ESRelationDoc = {
      id: relId,
      from_id: fromMemoryId,
      to_id: toMemoryId,
      rel_type: relationshipType,
      strength: props.strength,
      confidence: props.confidence,
      context: props.context ?? null,
      evidence_count: props.evidence_count,
      valid_from: props.valid_from,
      valid_until: props.valid_until ?? null,
      recorded_at: props.recorded_at,
      invalidated_by: props.invalidated_by ?? null,
      properties: JSON.stringify(props),
    };

    const res = await this.request(
      "PUT",
      `/${this.relationshipsIndex}/_doc/${encodeURIComponent(relId)}`,
      doc,
      { refresh: "wait_for" }
    );

    if (!res.ok) {
      throw new DatabaseConnectionError(`Failed to create relationship: ${JSON.stringify(res.data)}`);
    }

    return relId;
  }

  async getRelatedMemories(
    memoryId: string,
    opts?: { relationshipTypes?: string[]; maxDepth?: number; limit?: number }
  ): Promise<[Memory, Relationship][]> {
    const maxDepth = Math.max(1, Math.min(Number(opts?.maxDepth ?? 2) || 2, 10));
    const relTypes = opts?.relationshipTypes;

    const visited = new Set<string>([memoryId]);
    const results: [Memory, Relationship][] = [];
    let currentLevel = [memoryId];

    for (let depth = 0; depth < maxDepth; depth++) {
      if (currentLevel.length === 0) break;

      const shouldClauses: any[] = [];
      for (const id of currentLevel) {
        shouldClauses.push({ term: { from_id: id } });
        shouldClauses.push({ term: { to_id: id } });
      }

      const query: any = {
        bool: {
          should: shouldClauses,
          minimum_should_match: 1,
        },
      };

      if (relTypes && relTypes.length > 0) {
        query.bool.filter = [{ terms: { rel_type: relTypes } }];
      }

      const relRes = await this.request<any>(
        "POST",
        `/${this.relationshipsIndex}/_search`,
        { query, size: 1000 }
      );

      if (!relRes.ok || !relRes.data?.hits?.hits) break;

      const nextLevel: string[] = [];

      for (const hit of relRes.data.hits.hits) {
        const row = hit._source as ESRelationDoc;
        const otherId = currentLevel.includes(row.from_id) ? row.to_id : row.from_id;

        if (visited.has(otherId)) continue;
        visited.add(otherId);
        nextLevel.push(otherId);

        const mem = await this.getMemory(otherId, false);
        if (!mem) continue;

        const relProps = createRelationshipProperties({
          strength: row.strength,
          confidence: row.confidence,
          context: row.context ?? undefined,
          evidence_count: row.evidence_count,
          valid_from: row.valid_from,
          valid_until: row.valid_until ?? undefined,
          recorded_at: row.recorded_at,
          invalidated_by: row.invalidated_by ?? undefined,
        });

        const rel: Relationship = {
          id: row.id,
          from_memory_id: row.from_id,
          to_memory_id: row.to_id,
          type: row.rel_type,
          properties: relProps,
          description: undefined,
          bidirectional: false,
        };

        results.push([mem, rel]);
      }

      currentLevel = nextLevel;
    }

    if (opts?.limit !== undefined) {
      return results.slice(0, Math.max(0, Math.trunc(opts.limit)));
    }
    return results;
  }

  async getRelationshipsSince(since: Date): Promise<Relationship[]> {
    const query = {
      query: {
        range: {
          recorded_at: { gte: since.toISOString() },
        },
      },
      size: 1000,
    };

    const res = await this.request<any>(
      "POST",
      `/${this.relationshipsIndex}/_search`,
      query
    );

    if (!res.ok || !res.data?.hits?.hits) return [];

    return res.data.hits.hits.map((hit: any) => {
      const row = hit._source as ESRelationDoc;
      return {
        id: row.id,
        from_memory_id: row.from_id,
        to_memory_id: row.to_id,
        type: row.rel_type,
        properties: createRelationshipProperties({
          strength: row.strength,
          confidence: row.confidence,
          context: row.context ?? undefined,
          evidence_count: row.evidence_count,
          valid_from: row.valid_from,
          valid_until: row.valid_until ?? undefined,
          recorded_at: row.recorded_at,
          invalidated_by: row.invalidated_by ?? undefined,
        }),
        bidirectional: false,
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Search & Recall
  // ---------------------------------------------------------------------------

  async searchMemories(searchQuery: SearchQuery): Promise<Memory[]> {
    const filterClauses: any[] = [];
    const shouldClauses: any[] = [];
    const mustClauses: any[] = [];

    if (searchQuery.query) {
      const q = searchQuery.query.replace(/[\u2010-\u2015]/g, "-").trim();
      // Always run BM25 across all text fields (title, summary, content, tags)
      shouldClauses.push({
        multi_match: {
          query: q,
          fields: ["title^4", "summary^2", "content", "tags^3"],
          fuzziness: searchQuery.search_tolerance === "fuzzy" ? "AUTO" : undefined,
          minimum_should_match: "2<75%",
        },
      });

      // If semantic vector search is enabled, blend in semantic vector similarity boosted to match BM25 magnitude
      if (this._hasSemanticSupport) {
        shouldClauses.push({
          semantic: {
            field: "content.semantic",
            query: q,
            boost: 10.0,
          },
        });
        shouldClauses.push({
          semantic: {
            field: "summary.semantic",
            query: q,
            boost: 5.0,
          },
        });
      }
    }

    if (searchQuery.memory_types && searchQuery.memory_types.length > 0) {
      filterClauses.push({ terms: { type: searchQuery.memory_types } });
    }

    if (searchQuery.tags && searchQuery.tags.length > 0) {
      filterClauses.push({ terms: { tags: searchQuery.tags } });
    }

    if (searchQuery.project_path) {
      filterClauses.push({ term: { context_project_path: searchQuery.project_path } });
    }

    if (searchQuery.min_importance !== undefined) {
      filterClauses.push({ range: { importance: { gte: searchQuery.min_importance } } });
    }

    if (searchQuery.min_confidence !== undefined) {
      filterClauses.push({ range: { confidence: { gte: searchQuery.min_confidence } } });
    }

    if (searchQuery.created_after) {
      filterClauses.push({ range: { created_at: { gte: searchQuery.created_after.toISOString() } } });
    }

    if (searchQuery.created_before) {
      filterClauses.push({ range: { created_at: { lte: searchQuery.created_before.toISOString() } } });
    }

    const boolQuery: any = {};
    if (mustClauses.length > 0) boolQuery.must = mustClauses;
    if (shouldClauses.length > 0) {
      boolQuery.should = shouldClauses;
      boolQuery.minimum_should_match = 1;
    }
    if (filterClauses.length > 0) boolQuery.filter = filterClauses;

    const esSearchBody: any = {
      query: Object.keys(boolQuery).length > 0 ? { bool: boolQuery } : { match_all: {} },
      from: searchQuery.offset ?? 0,
      size: searchQuery.limit ?? 50,
      sort: searchQuery.query
        ? [{ _score: "desc" }, { importance: "desc" }, { created_at: "desc" }]
        : [{ importance: "desc" }, { created_at: "desc" }],
    };

    const res = await this.request<any>(
      "POST",
      `/${this.memoriesIndex}/_search`,
      esSearchBody
    );

    if (!res.ok || !res.data?.hits?.hits) return [];

    return res.data.hits.hits.map((hit: any) => this.mapSourceToMemory(hit._source));
  }

  async recallMemories(
    query: string,
    opts?: { memoryTypes?: string[]; projectPath?: string; limit?: number }
  ): Promise<Memory[]> {
    const filterClauses: any[] = [];
    const shouldClauses: any[] = [];

    if (opts?.memoryTypes && opts.memoryTypes.length > 0) {
      filterClauses.push({ terms: { type: opts.memoryTypes } });
    }
    if (opts?.projectPath) {
      filterClauses.push({ term: { context_project_path: opts.projectPath } });
    }

    if (query) {
      const q = query.replace(/[\u2010-\u2015]/g, "-").trim();
      shouldClauses.push({
        multi_match: {
          query: q,
          fields: ["title^4", "summary^2", "content"],
          minimum_should_match: "2<75%",
        },
      });
      if (this._hasSemanticSupport) {
        shouldClauses.push({
          semantic: {
            field: "content.semantic",
            query: q,
            boost: 10.0,
          },
        });
      }
    }

    const baseQuery = {
      bool: {
        must: query ? shouldClauses : [{ match_all: {} }],
        filter: filterClauses.length > 0 ? filterClauses : undefined,
      },
    };

    // M1 composite ranking with function_score & recency decay
    const esBody = {
      query: {
        function_score: {
          query: baseQuery,
          functions: [
            {
              field_value_factor: {
                field: "importance",
                factor: 0.4,
                missing: 0.5,
              },
            },
            {
              field_value_factor: {
                field: "confidence",
                factor: 0.2,
                missing: 0.8,
              },
            },
            {
              exp: {
                created_at: {
                  scale: "30d",
                  decay: 0.5,
                },
              },
            },
          ],
          score_mode: "sum",
          boost_mode: "multiply",
        },
      },
      size: opts?.limit ?? 20,
    };

    const res = await this.request<any>("POST", `/${this.memoriesIndex}/_search`, esBody);
    if (!res.ok || !res.data?.hits?.hits) {
      // Fallback to standard search if function_score is unsupported
      return this.searchMemories({
        query,
        terms: [],
        memory_types: opts?.memoryTypes ?? [],
        tags: [],
        project_path: opts?.projectPath,
        languages: [],
        frameworks: [],
        limit: opts?.limit ?? 20,
      });
    }

    return res.data.hits.hits.map((hit: any) => {
      const mem = this.mapSourceToMemory(hit._source);
      mem.match_info = {
        score: hit._score,
        match_quality: hit._score > 1.0 ? "high" : "medium",
      };
      return mem;
    });
  }

  // ---------------------------------------------------------------------------
  // Statistics & Temporal Versioning
  // ---------------------------------------------------------------------------

  async getMemoryStatistics(): Promise<Record<string, unknown>> {
    const body = {
      size: 0,
      aggs: {
        types: { terms: { field: "type", size: 50 } },
        avg_importance: { avg: { field: "importance" } },
        avg_confidence: { avg: { field: "confidence" } },
      },
    };

    const res = await this.request<any>("POST", `/${this.memoriesIndex}/_search`, body);
    const relCountRes = await this.request<any>("GET", `/${this.relationshipsIndex}/_count`);

    const totalCount = res.data?.hits?.total?.value ?? 0;
    const relCount = relCountRes.data?.count ?? 0;
    const agg = res.data?.aggregations;

    const memoriesByType: Record<string, number> = {};
    if (agg?.types?.buckets) {
      for (const bucket of agg.types.buckets) {
        memoriesByType[bucket.key] = bucket.doc_count;
      }
    }

    return {
      total_memories: { count: totalCount },
      total_relationships: { count: relCount },
      memories_by_type: memoriesByType,
      avg_importance: { avg_importance: agg?.avg_importance?.value ?? 0 },
      avg_confidence: { avg_confidence: agg?.avg_confidence?.value ?? 0 },
    };
  }

  async getMemoryStateAt(memoryId: string, timestamp: Date): Promise<Memory | null> {
    const body = {
      query: {
        bool: {
          must: [
            { term: { memory_id: memoryId } },
            { range: { recorded_at: { lte: timestamp.toISOString() } } },
          ],
        },
      },
      sort: [{ recorded_at: "desc" }],
      size: 1,
    };

    const res = await this.request<any>("POST", `/${this.versionsIndex}/_search`, body);
    if (res.ok && res.data?.hits?.hits?.length > 0) {
      const payload = res.data.hits.hits[0]._source.payload;
      if (payload) {
        return JSON.parse(payload) as Memory;
      }
    }

    // Fall back to current state if no snapshot prior to timestamp exists
    return this.getMemory(memoryId, false);
  }

  async getMemoryVersions(memoryId: string): Promise<Memory[]> {
    const body = {
      query: {
        term: { memory_id: memoryId },
      },
      sort: [{ recorded_at: "desc" }],
      size: 100,
    };

    const res = await this.request<any>("POST", `/${this.versionsIndex}/_search`, body);
    if (!res.ok || !res.data?.hits?.hits) return [];

    return res.data.hits.hits
      .map((h: any) => {
        try {
          return JSON.parse(h._source.payload) as Memory;
        } catch {
          return null;
        }
      })
      .filter((m: any): m is Memory => m !== null);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private mapSourceToMemory(src: any): Memory {
    return createMemory({
      id: src.id,
      type: src.type as MemoryType,
      title: src.title,
      content: typeof src.content === "string" ? src.content : src.content?.text ?? "",
      summary: typeof src.summary === "string" ? src.summary : src.summary?.text ?? undefined,
      tags: Array.isArray(src.tags) ? src.tags : [],
      importance: src.importance ?? 0.5,
      confidence: src.confidence ?? 0.8,
      effectiveness: src.effectiveness ?? 0.5,
      usage_count: src.usage_count ?? 0,
      created_at: new Date(src.created_at),
      updated_at: new Date(src.updated_at ?? src.created_at),
      last_accessed: src.last_accessed ? new Date(src.last_accessed) : undefined,
      valid_from: src.valid_from ? new Date(src.valid_from) : undefined,
      valid_until: src.valid_until ? new Date(src.valid_until) : undefined,
      recorded_at: src.recorded_at ? new Date(src.recorded_at) : undefined,
      invalidated_by: src.invalidated_by ?? undefined,
      context_project_path: src.context_project_path ?? undefined,
      context_branch: src.context_branch ?? undefined,
      context_language: src.context_language ?? undefined,
      context_framework: src.context_framework ?? undefined,
      context_summary: src.context_summary ?? undefined,
      metadata: src.metadata ?? {},
    });
  }
}
