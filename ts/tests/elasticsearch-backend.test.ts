/**
 * Unit tests for ElasticsearchBackend with mock fetch.
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { ElasticsearchBackend } from "../src/backends/elasticsearch.ts";
import { createMemory } from "../src/models.ts";

describe("ElasticsearchBackend (Mocked)", () => {
  const originalFetch = globalThis.fetch;
  let mockRequests: { url: string; method: string; body?: any }[] = [];
  let mockResponses: Record<string, { status: number; ok: boolean; data: any }> = {};

  beforeEach(() => {
    mockRequests = [];
    mockResponses = {};

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      mockRequests.push({ url, method, body });

      // Match response by path / pattern
      for (const [pattern, res] of Object.entries(mockResponses)) {
        if (url.includes(pattern)) {
          return new Response(JSON.stringify(res.data), {
            status: res.status,
            headers: { "Content-Type": "application/json" },
          });
        }
      }

      // Default 200 OK
      return new Response(JSON.stringify({ acknowledged: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("connect verifies cluster root info", async () => {
    mockResponses["/"] = {
      status: 200,
      ok: true,
      data: {
        version: { number: "9.6.0", build_flavor: "serverless" },
      },
    };

    const backend = new ElasticsearchBackend({
      url: "https://test-es.cloud:443",
      apiKey: "test-api-key",
    });

    const connected = await backend.connect();
    expect(connected).toBe(true);
    expect(mockRequests.length).toBeGreaterThanOrEqual(1);
    expect(mockRequests[0].url).toBe("https://test-es.cloud:443/");
  });

  test("initializeSchema creates indices idempotently", async () => {
    // Return 404 for HEAD checks
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? "GET";
      mockRequests.push({ url, method });

      if (method === "HEAD") {
        return new Response(null, { status: 404 });
      }
      return new Response(JSON.stringify({ acknowledged: true }), { status: 200 });
    };

    const backend = new ElasticsearchBackend({
      url: "https://test-es.cloud:443",
      indexPrefix: "test_mg",
    });

    await backend.initializeSchema();

    const putRequests = mockRequests.filter((r) => r.method === "PUT");
    expect(putRequests.length).toBe(3); // memories, relationships, versions
    expect(putRequests[0].url).toContain("test_mg_memories");
    expect(putRequests[1].url).toContain("test_mg_relationships");
    expect(putRequests[2].url).toContain("test_mg_versions");
  });

  test("storeMemory and getMemory roundtrip", async () => {
    const memory = createMemory({
      id: "mem-test-1",
      type: "solution",
      title: "Fix Docker socket permissions",
      content: "Use usermod -aG docker to allow non-root access",
      tags: ["docker", "permissions"],
      importance: 0.9,
    });

    const createdAtStr = typeof memory.created_at === "string" ? memory.created_at : memory.created_at.toISOString();
    const updatedAtStr = typeof memory.updated_at === "string" ? memory.updated_at : memory.updated_at.toISOString();

    mockResponses["test_mg_memories/_doc/mem-test-1"] = {
      status: 200,
      ok: true,
      data: {
        _source: {
          id: memory.id,
          type: memory.type,
          title: memory.title,
          content: memory.content,
          tags: memory.tags,
          importance: memory.importance,
          confidence: memory.confidence,
          created_at: createdAtStr,
          updated_at: updatedAtStr,
        },
      },
    };

    mockResponses["test_mg_relationships/_search"] = {
      status: 200,
      ok: true,
      data: {
        hits: {
          hits: [
            {
              _source: {
                from_id: "mem-test-1",
                to_id: "mem-problem-0",
                rel_type: "SOLVES",
              },
            },
          ],
        },
      },
    };

    const backend = new ElasticsearchBackend({
      url: "https://test-es.cloud:443",
      indexPrefix: "test_mg",
    });

    const storedId = await backend.storeMemory(memory);
    expect(storedId).toBe("mem-test-1");

    const retrieved = await backend.getMemory("mem-test-1", true);
    expect(retrieved).not.toBeNull();
    expect(retrieved?.id).toBe("mem-test-1");
    expect(retrieved?.title).toBe("Fix Docker socket permissions");
    expect(retrieved?.relationships?.["SOLVES"]).toContain("mem-problem-0");
  });

  test("searchMemories builds correct multi_match query", async () => {
    mockResponses["test_mg_memories/_search"] = {
      status: 200,
      ok: true,
      data: {
        hits: {
          hits: [
            {
              _source: {
                id: "mem-search-1",
                type: "code_pattern",
                title: "Memory leak in event emitter",
                content: "Remember to remove listeners on unsubscribe",
                tags: ["memory", "leak"],
                importance: 0.8,
                confidence: 0.9,
                created_at: new Date().toISOString(),
              },
            },
          ],
        },
      },
    };

    const backend = new ElasticsearchBackend({
      url: "https://test-es.cloud:443",
      indexPrefix: "test_mg",
    });

    const results = await backend.searchMemories({
      query: "event emitter leak",
      terms: [],
      memory_types: ["code_pattern"],
      tags: ["memory"],
      languages: [],
      frameworks: [],
      limit: 10,
    });

    expect(results.length).toBe(1);
    expect(results[0].title).toBe("Memory leak in event emitter");

    const searchCall = mockRequests.find(
      (r) => r.method === "POST" && r.url.includes("test_mg_memories/_search")
    );
    expect(searchCall).toBeDefined();
    expect(searchCall?.body?.query?.bool?.should).toBeDefined();
    expect(searchCall?.body?.query?.bool?.filter).toBeDefined();
  });

  test("getRelatedMemories traverses BFS graph", async () => {
    const mem1 = createMemory({ id: "node-1", type: "problem", title: "Node 1", content: "..." });
    const mem2 = createMemory({ id: "node-2", type: "solution", title: "Node 2", content: "..." });

    mockResponses["test_mg_relationships/_search"] = {
      status: 200,
      ok: true,
      data: {
        hits: {
          hits: [
            {
              _source: {
                id: "rel-1-2",
                from_id: "node-1",
                to_id: "node-2",
                rel_type: "SOLVES",
                strength: 0.95,
                confidence: 0.9,
                recorded_at: new Date().toISOString(),
                valid_from: new Date().toISOString(),
                evidence_count: 1,
              },
            },
          ],
        },
      },
    };

    const mem2CreatedAt = typeof mem2.created_at === "string" ? mem2.created_at : mem2.created_at.toISOString();
    const mem2UpdatedAt = typeof mem2.updated_at === "string" ? mem2.updated_at : mem2.updated_at.toISOString();

    mockResponses["test_mg_memories/_doc/node-2"] = {
      status: 200,
      ok: true,
      data: {
        _source: {
          id: mem2.id,
          type: mem2.type,
          title: mem2.title,
          content: mem2.content,
          tags: [],
          importance: 0.5,
          confidence: 0.8,
          created_at: mem2CreatedAt,
          updated_at: mem2UpdatedAt,
        },
      },
    };

    const backend = new ElasticsearchBackend({
      url: "https://test-es.cloud:443",
      indexPrefix: "test_mg",
    });

    const related = await backend.getRelatedMemories("node-1", { maxDepth: 1 });
    expect(related.length).toBe(1);
    const [relatedMem, rel] = related[0];
    expect(relatedMem.id).toBe("node-2");
    expect(rel.type).toBe("SOLVES");
    expect(rel.properties.strength).toBe(0.95);
  });
});
