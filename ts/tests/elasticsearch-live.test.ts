/**
 * Live integration test against Elastic Cloud Serverless.
 *
 * Runs only when MEMORY_ELASTICSEARCH_URL and MEMORY_ELASTICSEARCH_API_KEY are configured.
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { ElasticsearchBackend } from "../src/backends/elasticsearch.ts";
import { createMemory } from "../src/models.ts";

const ES_URL = process.env.MEMORY_ELASTICSEARCH_URL;
const ES_API_KEY = process.env.MEMORY_ELASTICSEARCH_API_KEY;

const isLive = Boolean(ES_URL && ES_API_KEY);
const describeLive = isLive ? describe : describe.skip;

describeLive("Elasticsearch Live Integration (Elastic Cloud)", () => {
  const testPrefix = `mg_test_${Date.now()}`;
  let backend: ElasticsearchBackend;

  beforeAll(async () => {
    backend = new ElasticsearchBackend({
      url: ES_URL,
      apiKey: ES_API_KEY,
      indexPrefix: testPrefix,
      timeout: 15000,
    });

    await backend.connect();
    await backend.initializeSchema();
  });

  afterAll(async () => {
    if (backend) {
      // Clean up test indices
      try {
        await (backend as any).request("DELETE", `/${testPrefix}_memories`);
        await (backend as any).request("DELETE", `/${testPrefix}_relationships`);
        await (backend as any).request("DELETE", `/${testPrefix}_versions`);
      } catch {
        // ignore cleanup errors
      }
      await backend.disconnect();
    }
  });

  test("healthCheck returns cluster status", async () => {
    const health = await backend.healthCheck();
    expect(health.connected).toBe(true);
    expect(health.backend_type).toBe("elasticsearch");
  });

  test(
    "CRUD, relationships, hybrid semantic search, and traversal lifecycle",
    async () => {
      // 1. Store a problem memory
    const problemMem = createMemory({
      id: `prob-${Date.now()}`,
      type: "problem",
      title: "Docker daemon socket permission denied",
      content: "EACCES: permission denied, open /var/run/docker.sock when running docker as non-root",
      tags: ["docker", "permissions", "eacces"],
      importance: 0.9,
      confidence: 0.95,
    });
    await backend.storeMemory(problemMem);

    // 2. Store a solution memory
    const solutionMem = createMemory({
      id: `sol-${Date.now()}`,
      type: "solution",
      title: "Add non-root user to docker group",
      content: "Run sudo usermod -aG docker $USER and restart session to access the daemon socket without root",
      tags: ["docker", "solution", "group"],
      importance: 0.85,
      confidence: 0.9,
    });
    await backend.storeMemory(solutionMem);

    // 3. Create relationship: solution SOLVES problem
    const relId = await backend.createRelationship(
      solutionMem.id!,
      problemMem.id!,
      "SOLVES",
      { strength: 0.95, confidence: 0.9 }
    );
    expect(relId).toBeDefined();

    // 4. Retrieve solution memory and verify relationships attached
    const retrievedSol = await backend.getMemory(solutionMem.id!, true);
    expect(retrievedSol).not.toBeNull();
    expect(retrievedSol?.relationships?.["SOLVES"]).toContain(problemMem.id!);

    // 5. Semantic / Hybrid search
    // Query with DIFFERENT phrasing to test semantic retrieval
    const searchResults = await backend.searchMemories({
      query: "container permission socket non-root",
      terms: [],
      memory_types: [],
      tags: [],
      languages: [],
      frameworks: [],
      limit: 10,
    });

    expect(searchResults.length).toBeGreaterThan(0);
    const foundIds = searchResults.map((m) => m.id);
    expect(foundIds.includes(problemMem.id!) || foundIds.includes(solutionMem.id!)).toBe(true);

    // 6. Recall memories (composite ranking)
    const recallResults = await backend.recallMemories("docker permissions", { limit: 5 });
    expect(recallResults.length).toBeGreaterThan(0);

    // 7. Graph BFS traversal
    const related = await backend.getRelatedMemories(solutionMem.id!, { maxDepth: 1 });
    expect(related.length).toBe(1);
    const [targetMem, rel] = related[0];
    expect(targetMem.id).toBe(problemMem.id!);
    expect(rel.type).toBe("SOLVES");

    // 8. Statistics
    const stats = await backend.getMemoryStatistics();
    const totalMem = stats["total_memories"] as { count: number };
    const totalRel = stats["total_relationships"] as { count: number };
    expect(totalMem.count).toBeGreaterThanOrEqual(2);
    expect(totalRel.count).toBeGreaterThanOrEqual(1);

    // 9. Update memory and check version snapshot
    await backend.updateMemory({
      ...solutionMem,
      title: "Add non-root user to docker group (Updated)",
    });

    const versions = await backend.getMemoryVersions(solutionMem.id!);
    expect(versions.length).toBeGreaterThanOrEqual(1);

    // 10. Delete memory
    const deleted = await backend.deleteMemory(solutionMem.id!);
    expect(deleted).toBe(true);

    const postDelete = await backend.getMemory(solutionMem.id!, false);
    expect(postDelete).toBeNull();
  }, 30000);
});
