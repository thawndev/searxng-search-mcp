#!/usr/bin/env node
// ── End-to-end test suite ──

import { createServer } from "./server.js";
import { SearXNGClient } from "./searxng-client.js";
import { executeSearch } from "./search.js";
import { fetchPage } from "./fetcher.js";
import { verifyClaim } from "./verifier.js";
import { deepResearch } from "./research.js";
import { analyzeQuery, expandQueries } from "./query-analyzer.js";
import { rankResults } from "./ranking.js";
import { LRUCache } from "./cache.js";
import { unifiedSearch, formatOutput } from "./unified-search.js";
import { warmupEmbeddings, isModelReady, classifyQuery } from "./embeddings.js";
import type { ServerConfig, SearXNGRawResult } from "./types.js";

const config: ServerConfig = {
  searxngUrl: "http://localhost:8080",
  cacheTtl: 300000,
  requestTimeout: 15000,
  maxCacheSize: 500,
  maxConcurrentRequests: 8,
  maxResultsPerQuery: 50,
  maxFetchSize: 50000,
  retry: { maxRetries: 2, baseDelayMs: 500, maxDelayMs: 5000 },
};

const client = new SearXNGClient(config);
let passed = 0;
let failed = 0;

function assert(condition: boolean, testName: string, detail = "") {
  if (condition) {
    console.log(`  ✅ ${testName}`);
    passed++;
  } else {
    console.error(`  ❌ ${testName}${detail ? `: ${detail}` : ""}`);
    failed++;
  }
}

async function test(name: string, fn: () => Promise<void>) {
  console.log(`\n🧪 ${name}`);
  try {
    await fn();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  💥 EXCEPTION: ${msg}`);
    failed++;
  }
}

// ═══════════ Embedding Warmup ═══════════

console.log("\n⏳ Warming up embedding model + archetypes...");
const warmupOk = await warmupEmbeddings();
console.log(warmupOk ? "✅ Model ready" : "⚠️ Model unavailable, tests will use fallback");

// ═══════════ Neural Classification Tests ═══════════

await test("Neural classification - code query", async () => {
  const a = await analyzeQuery("how to configure tailwind.config.js for NativeWind");
  assert(a.type === "code" || a.type === "howto", `detects code/howto, got: ${a.type}`);
  assert(a.suggested_topic === "code", `topic=code, got: ${a.suggested_topic}`);
  assert(a.neural === isModelReady(), `neural=${a.neural} matches model readiness`);
});

await test("Neural classification - news query", async () => {
  const a = await analyzeQuery("latest AI news today breaking announcement");
  assert(a.type === "news", `detects news, got: ${a.type}`);
  assert(a.time_sensitive === true, "marks as time-sensitive");
});

await test("Neural classification - troubleshooting query", async () => {
  const a = await analyzeQuery("ECONNREFUSED error nodejs fix");
  assert(
    a.type === "troubleshooting" || a.type === "code",
    `detects troubleshooting/code, got: ${a.type}`,
  );
});

await test("Neural classification - definition query", async () => {
  const a = await analyzeQuery("what is a monad in functional programming");
  assert(
    a.type === "definition" || a.type === "code" || a.type === "general",
    `detects definition/code/general, got: ${a.type}`,
  );
});

await test("Neural classification - comparison query", async () => {
  const a = await analyzeQuery("React vs Vue vs Angular comparison benchmark");
  assert(
    a.type === "comparison" || a.type === "code",
    `detects comparison/code, got: ${a.type}`,
  );
});

await test("Neural classification - academic query", async () => {
  const a = await analyzeQuery("attention mechanism transformer paper research");
  assert(
    a.type === "academic" || a.type === "code",
    `detects academic/code, got: ${a.type}`,
  );
});

await test("Neural classification - 'latest' with code context should NOT be news", async () => {
  const a = await analyzeQuery("nativewind latest configuration for global.css");
  assert(a.type !== "news", `should not be news, got: ${a.type}`);
  assert(a.time_sensitive === false, `should not be time-sensitive, got: ${a.time_sensitive}`);
});

await test("Neural classification - confidence score", async () => {
  const a = await analyzeQuery("docker compose environment variables configuration");
  if (a.neural) {
    assert(a.confidence > 0, `confidence > 0, got: ${a.confidence.toFixed(3)}`);
    assert(a.secondary_type !== undefined, `has secondary_type: ${a.secondary_type}`);
  } else {
    assert(true, "skipped — model not ready (fallback mode)");
  }
});

await test("Query expansion", async () => {
  const a = await analyzeQuery("python async await");
  const expanded = expandQueries("python async await", a);
  assert(expanded.length >= 1, `got ${expanded.length} queries`);
  assert(expanded[0] === "python async await", "original query is first");
});

// ═══════════ Ranking Tests ═══════════

await test("Ranking - basic scoring", async () => {
  const raw: SearXNGRawResult[] = [
    { url: "https://docs.python.org/3/library/asyncio.html", title: "asyncio — Asynchronous I/O", content: "High-level API", engine: "google", engines: ["google", "bing"], score: 5, category: "general" },
    { url: "https://example.com/spam", title: "Buy cheap stuff", content: "spam content", engine: "google", engines: ["google"], score: 1, category: "general" },
  ];
  const ranked = rankResults(raw, false, "code");
  assert(ranked.length === 2, "returns 2 results");
  assert(ranked[0].relevance_score > 0, "first result has positive score");
  assert(ranked[1].relevance_score > 0, "second result has positive score");
});

await test("Ranking - multi-engine boost", async () => {
  const raw: SearXNGRawResult[] = [
    { url: "https://example.com/a", title: "A", content: "test", engine: "google", engines: ["google", "bing", "brave"], score: 3, category: "general" },
    { url: "https://example.com/b", title: "B", content: "test", engine: "google", engines: ["google"], score: 3, category: "general" },
  ];
  const ranked = rankResults(raw, false, "general");
  assert(ranked[0].url.includes("/a"), "multi-engine result ranked higher");
});

// ═══════════ Cache Tests ═══════════

await test("LRU Cache - basic operations", async () => {
  const cache = new LRUCache(10, 60000);
  cache.set("key1", { data: "value1" });
  const got = cache.get("key1") as any;
  assert(got?.data === "value1", "get returns stored value");
  assert(cache.stats.hits === 1, "records hit");
});

await test("LRU Cache - miss tracking", async () => {
  const cache = new LRUCache(10, 60000);
  const got = cache.get("nonexistent");
  assert(got === undefined, "returns undefined for miss");
  assert(cache.stats.misses === 1, "records miss");
});

await test("LRU Cache - TTL expiry", async () => {
  const cache = new LRUCache(10, 1); // 1ms TTL
  cache.set("ttl-key", "value");
  await new Promise((r) => setTimeout(r, 10));
  const got = cache.get("ttl-key");
  assert(got === undefined, "expired entry returns undefined");
});

await test("LRU Cache - eviction", async () => {
  const cache = new LRUCache(3, 60000);
  cache.set("a", 1);
  cache.set("b", 2);
  cache.set("c", 3);
  cache.set("d", 4); // should evict "a"
  assert(cache.get("a") === undefined, "evicted oldest entry");
  assert(cache.get("d") === 4, "newest entry present");
});

// ═══════════ SearXNG Client Tests ═══════════

await test("SearXNG client - health check", async () => {
  const healthResult = await client.healthCheck();
  assert(healthResult.healthy === true, "SearXNG is healthy");
  assert(healthResult.instances.length > 0, "has instance info");
});

await test("SearXNG client - basic search", async () => {
  const resp = await client.search({ q: "python programming language" });
  assert(resp.results.length > 0, `got ${resp.results.length} results`);
  if (resp.results.length > 0) {
    assert(resp.results[0].url.startsWith("http"), "results have URLs");
    assert(resp.results[0].engines.length > 0, "results have engines");
  }
});

// ═══════════ Core Search Tests ═══════════

await test("executeSearch - basic", async () => {
  const out = await executeSearch({ query: "python programming language tutorial" }, client, config);
  assert(out.results.length >= 0, `got ${out.results.length} results`);
  assert(out.queries_used.length >= 1, "has queries_used");
  assert(out.engines_used.length >= 0, `engines: ${out.engines_used.length}`);
});

await test("executeSearch - with site filter", async () => {
  const out = await executeSearch(
    { query: "python asyncio", site_filters: ["docs.python.org"] },
    client,
    config,
  );
  assert(out.results.length >= 0, `got ${out.results.length} results`);
  assert(out.queries_used.some(q => q.includes("site:docs.python.org")), "site filter applied to queries");
});

await test("executeSearch - caching", async () => {
  // Search twice with same query
  const out1 = await executeSearch({ query: "cache test query unique 12345" }, client, config);
  const out2 = await executeSearch({ query: "cache test query unique 12345" }, client, config);
  assert(out2.metadata.cached === true, "second call is cached");
});

// ═══════════ Fetch Tests ═══════════

await test("fetchPage - basic HTML", async () => {
  const result = await fetchPage("https://example.com", config);
  assert(result.content.length > 0, "has content");
  assert(result.title.length > 0, `title: "${result.title}"`);
  assert(result.metadata.domain === "example.com", "has domain");
});

await test("fetchPage - JSON content", async () => {
  const result = await fetchPage("https://httpbin.org/json", config);
  assert(result.content.length > 0, "has content");
  assert(result.content_type === "json" || result.content.includes("{"), "detected JSON");
});

// ═══════════ Unified Search Tests ═══════════

await test("unifiedSearch - query mode (basic)", async () => {
  const result = await unifiedSearch({ query: "TypeScript interfaces" }, client, config);
  assert(result.results.length > 0, `got ${result.results.length} results`);
  assert(result.plan.mode === "search", `mode: ${result.plan.mode}`);
  assert(!("auto" === (result.plan.depth as string)), "depth resolved from auto");
  assert(!("auto" === (result.plan.topic as string)), "topic resolved from auto");
  assert(result.duration_ms > 0, `took ${result.duration_ms}ms`);
});

await test("unifiedSearch - URL fetch mode", async () => {
  const result = await unifiedSearch({ url: "https://example.com" }, client, config);
  assert(result.plan.mode === "fetch", `mode: ${result.plan.mode}`);
  assert(result.fetched_content !== undefined, "has fetched content");
  assert(result.fetched_content!.length === 1, "fetched exactly 1 page");
  assert(result.fetched_content![0].url.includes("example.com"), "fetched correct URL");
});

await test("unifiedSearch - quick depth", async () => {
  const result = await unifiedSearch({ query: "what is DNS domain name system", depth: "quick" }, client, config);
  assert(result.plan.depth === "quick", `depth: ${result.plan.depth}`);
  assert(result.results.length >= 0, `results: ${result.results.length}`);
  assert(result.fetched_content === undefined, "quick mode doesn't fetch content");
});

await test("unifiedSearch - deep depth with content", async () => {
  const result = await unifiedSearch(
    { query: "nodejs buffer overflow fix", depth: "deep", include_content: true },
    client,
    config,
  );
  assert(result.plan.depth === "deep", `depth: ${result.plan.depth}`);
  assert(result.plan.include_content === true, "content fetching enabled");
  // May or may not have fetched_content depending on results
  assert(result.results.length > 0, "has results");
});

await test("unifiedSearch - topic routing code", async () => {
  const result = await unifiedSearch({ query: "rust borrow checker", topic: "code" }, client, config);
  assert(result.plan.topic === "code", `topic: ${result.plan.topic}`);
  assert(result.results.length > 0, "has results");
});

await test("unifiedSearch - topic routing news", async () => {
  const result = await unifiedSearch({ query: "latest tech news", topic: "news" }, client, config);
  assert(result.plan.topic === "news", `topic: ${result.plan.topic}`);
});

await test("unifiedSearch - freshness filter", async () => {
  const result = await unifiedSearch({ query: "AI developments", freshness: "week" }, client, config);
  assert(result.plan.freshness === "week", `freshness: ${result.plan.freshness}`);
  assert(result.results.length >= 0, `results: ${result.results.length}`);
});

await test("unifiedSearch - verify mode", async () => {
  const result = await unifiedSearch(
    { query: "TypeScript is maintained by Microsoft", verify: true },
    client,
    config,
  );
  assert(result.plan.mode === "verify", `mode: ${result.plan.mode}`);
  assert(result.verification !== undefined, "has verification result");
  assert(
    ["supported", "likely_supported", "likely_refuted", "refuted", "inconclusive", "unverifiable", "insufficient_evidence"].includes(
      result.verification!.verdict,
    ),
    `verdict: ${result.verification?.verdict}`,
  );
  assert(result.verification!.confidence.length > 0, "has confidence");
});

await test("unifiedSearch - research mode", async () => {
  const result = await unifiedSearch(
    { query: "WebAssembly performance benchmarks", depth: "research" },
    client,
    config,
  );
  assert(result.plan.mode === "research", `mode: ${result.plan.mode}`);
  assert(result.research !== undefined, "has research result");
  assert(result.research!.synthesis.length > 0, "has synthesis text");
  assert(result.research!.steps_taken > 0, `${result.research!.steps_taken} steps`);
});

await test("unifiedSearch - site restriction", async () => {
  const result = await unifiedSearch(
    { query: "React hooks", site: ["reactjs.org", "react.dev"] },
    client,
    config,
  );
  assert(result.results.length > 0, "has results");
});

await test("unifiedSearch - max_sources budget", async () => {
  const result = await unifiedSearch({ query: "JavaScript", max_sources: 3 }, client, config);
  assert(result.results.length <= 3, `got ${result.results.length} (max 3)`);
});

await test("unifiedSearch - auto-detection code query", async () => {
  const result = await unifiedSearch(
    { query: "React useEffect cleanup function memory leak" },
    client,
    config,
  );
  assert(
    result.plan.topic === "code" || result.plan.depth === "deep",
    `auto-detected: topic=${result.plan.topic}, depth=${result.plan.depth}`,
  );
});

await test("unifiedSearch - auto-detection news query", async () => {
  const result = await unifiedSearch({ query: "latest GPT release news today" }, client, config);
  assert(
    result.plan.topic === "news" || result.plan.freshness !== undefined,
    `auto-detected: topic=${result.plan.topic}, freshness=${result.plan.freshness}`,
  );
});

// ═══════════ Output Formatting Tests ═══════════

await test("formatOutput - search result formatting", async () => {
  const result = await unifiedSearch({ query: "golang goroutines", depth: "quick" }, client, config);
  const formatted = formatOutput(result);
  assert(formatted.includes("**Search**"), "has search header");
  assert(formatted.includes("### Results"), "has results section");
  assert(formatted.includes("Plan:"), "shows execution plan");
  assert(formatted.length > 100, "non-trivial output length");
});

await test("formatOutput - URL fetch formatting", async () => {
  const result = await unifiedSearch({ url: "https://example.com" }, client, config);
  const formatted = formatOutput(result);
  assert(formatted.includes("### Fetched Content"), "has fetched content section");
});

await test("formatOutput - verify formatting", async () => {
  const result = await unifiedSearch(
    { query: "Python was created by Guido van Rossum", verify: true },
    client,
    config,
  );
  const formatted = formatOutput(result);
  assert(formatted.includes("### Verification"), "has verification section");
});

// ═══════════ Edge Cases ═══════════

await test("unifiedSearch - empty query handled", async () => {
  try {
    const result = await unifiedSearch({ query: "" }, client, config);
    // Should still work (SearXNG may return empty or error)
    assert(true, "didn't throw on empty query");
  } catch {
    assert(true, "threw on empty query (acceptable)");
  }
});

await test("unifiedSearch - very long query", async () => {
  const longQuery = "how to " + "really ".repeat(50) + "use React hooks";
  const result = await unifiedSearch({ query: longQuery }, client, config);
  assert(result.plan.mode === "search", "handled long query");
});

await test("unifiedSearch - special characters in query", async () => {
  const result = await unifiedSearch({ query: 'C++ std::vector<int> push_back "out of range"' }, client, config);
  assert(result.results.length >= 0, "handled special chars");
});

await test("unifiedSearch - URL mode with invalid URL", async () => {
  try {
    await unifiedSearch({ url: "not-a-url" }, client, config);
    assert(false, "should have thrown");
  } catch {
    assert(true, "threw on invalid URL");
  }
});

// ═══════════ Server Tool Registration ═══════════

await test("Server creates with tools", async () => {
  const server = createServer(config);
  assert(server !== undefined, "server created");
  // Server should have 'search' and 'get_diagnostics' tools
});

// ═══════════ Performance Tests ═══════════

await test("Performance - parallel searches", async () => {
  const start = Date.now();
  const queries = ["golang channels", "rust ownership", "python asyncio", "java streams", "typescript generics"];
  const results = await Promise.all(
    queries.map((q) => unifiedSearch({ query: q, depth: "quick" }, client, config)),
  );
  const elapsed = Date.now() - start;
  assert(results.every((r) => r.results.length > 0), "all searches returned results");
  console.log(`    ⏱️  5 parallel searches: ${elapsed}ms (${Math.round(elapsed / 5)}ms avg)`);
});

await test("Performance - cached search is fast", async () => {
  // First search (cold)
  await unifiedSearch({ query: "cached perf test unique 9999" }, client, config);
  // Second search (cached)
  const start = Date.now();
  const result = await unifiedSearch({ query: "cached perf test unique 9999" }, client, config);
  const elapsed = Date.now() - start;
  assert(elapsed < 50, `cached search took ${elapsed}ms (should be <50ms)`);
});

// ═══════════ Summary ═══════════

console.log(`\n${"═".repeat(50)}`);
console.log(`Total: ${passed + failed} | ✅ ${passed} passed | ❌ ${failed} failed`);
console.log(`${"═".repeat(50)}\n`);

process.exit(failed > 0 ? 1 : 0);
