#!/usr/bin/env node
// ── Eval Harness ──
// Tests neural classification accuracy and end-to-end search quality.

import { analyzeQuery } from "./query-analyzer.js";
import { warmupEmbeddings, isModelReady, classifyQuery } from "./embeddings.js";
import { unifiedSearch, formatOutput } from "./unified-search.js";
import { SearXNGClient } from "./searxng-client.js";
import type { ServerConfig, QueryType } from "./types.js";

const config: ServerConfig = {
  searxngUrl: process.env.SEARXNG_URL || "http://localhost:8080",
  cacheTtl: 300_000,
  requestTimeout: 15_000,
  maxCacheSize: 200,
  maxConcurrentRequests: 8,
  maxResultsPerQuery: 50,
  maxFetchSize: 50_000,
  retry: { maxRetries: 2, baseDelayMs: 500, maxDelayMs: 5000 },
};

const client = new SearXNGClient(config);

// ─── Classification Test Cases ───

interface ClassifyTestCase {
  query: string;
  expectedTypes: QueryType[];  // any of these is acceptable
  expectedTimeSensitive: boolean;
  expectedTopic: string;
  adversarial?: string;        // why this is tricky
}

const CLASSIFY_TESTS: ClassifyTestCase[] = [
  // Code queries
  { query: "nativewind latest configuration for global.css", expectedTypes: ["code", "howto", "technical"], expectedTimeSensitive: false, expectedTopic: "code", adversarial: "'latest' should NOT trigger news/time-sensitive for code queries" },
  { query: "React useEffect cleanup function memory leak", expectedTypes: ["code", "troubleshooting"], expectedTimeSensitive: false, expectedTopic: "code" },
  { query: "docker compose environment variables configuration", expectedTypes: ["code", "howto", "technical"], expectedTimeSensitive: false, expectedTopic: "code" },
  { query: "tailwind.config.js content array setup", expectedTypes: ["code", "howto", "technical"], expectedTimeSensitive: false, expectedTopic: "code", adversarial: "file extension should hint code" },
  { query: "prisma schema one-to-many relation", expectedTypes: ["code", "howto", "technical"], expectedTimeSensitive: false, expectedTopic: "code" },
  { query: "Error: Cannot find module '@/components/Button'", expectedTypes: ["code", "troubleshooting"], expectedTimeSensitive: false, expectedTopic: "code", adversarial: "error message should be troubleshooting/code" },

  // News queries
  { query: "latest AI news today", expectedTypes: ["news"], expectedTimeSensitive: true, expectedTopic: "news" },
  { query: "breaking tech announcement this week", expectedTypes: ["news"], expectedTimeSensitive: true, expectedTopic: "news" },

  // Academic queries
  { query: "attention mechanism transformer paper research", expectedTypes: ["academic", "code"], expectedTimeSensitive: false, expectedTopic: "academic" },

  // Definition queries
  { query: "what is a monad in functional programming", expectedTypes: ["definition", "code", "general"], expectedTimeSensitive: false, expectedTopic: "general" },

  // Comparison queries
  { query: "React vs Vue vs Angular comparison benchmark", expectedTypes: ["comparison", "code"], expectedTimeSensitive: false, expectedTopic: "code" },

  // How-to queries
  { query: "how to deploy Next.js app to Vercel", expectedTypes: ["howto", "code", "technical"], expectedTimeSensitive: false, expectedTopic: "code" },

  // Factual queries
  { query: "population of Tokyo Japan 2024", expectedTypes: ["factual", "general"], expectedTimeSensitive: false, expectedTopic: "general" },

  // Troubleshooting
  { query: "ECONNREFUSED error nodejs fix", expectedTypes: ["troubleshooting", "code"], expectedTimeSensitive: false, expectedTopic: "code" },
  { query: "why does my React component re-render infinitely", expectedTypes: ["troubleshooting", "code"], expectedTimeSensitive: false, expectedTopic: "code" },
];

// ─── Run Eval ───

async function runEval() {
  console.log("⏳ Warming up embeddings + archetypes...");
  const ok = await warmupEmbeddings();
  console.log(ok ? "✅ Model ready — neural classification active" : "⚠️ Model unavailable — fallback only");
  console.log("");

  // 1. Classification accuracy
  console.log("═══════ CLASSIFICATION EVAL ═══════\n");
  let classCorrect = 0;
  let classTotal = CLASSIFY_TESTS.length;
  let tsCorrect = 0;
  let topicCorrect = 0;

  for (const tc of CLASSIFY_TESTS) {
    const analysis = await analyzeQuery(tc.query);
    const typeOk = tc.expectedTypes.includes(analysis.type);
    const tsOk = analysis.time_sensitive === tc.expectedTimeSensitive;
    const topicOk = analysis.suggested_topic === tc.expectedTopic;

    if (typeOk) classCorrect++;
    if (tsOk) tsCorrect++;
    if (topicOk) topicCorrect++;

    const icon = typeOk && tsOk ? "✅" : typeOk ? "⚠️" : "❌";
    const neural = analysis.neural ? "neural" : "fallback";
    const confStr = analysis.confidence > 0 ? ` conf=${analysis.confidence.toFixed(3)}` : "";
    console.log(`${icon} "${tc.query.slice(0, 55)}..."`);
    console.log(`   type=${analysis.type} (expected: ${tc.expectedTypes.join("|")}) [${neural}${confStr}]`);
    if (!tsOk) console.log(`   ⚠️ time_sensitive=${analysis.time_sensitive} (expected: ${tc.expectedTimeSensitive})`);
    if (!topicOk) console.log(`   ⚠️ topic=${analysis.suggested_topic} (expected: ${tc.expectedTopic})`);
    if (tc.adversarial && !typeOk) console.log(`   💡 ${tc.adversarial}`);
  }

  console.log(`\nClassification: ${classCorrect}/${classTotal} (${((classCorrect / classTotal) * 100).toFixed(0)}%)`);
  console.log(`Time-sensitive: ${tsCorrect}/${classTotal} (${((tsCorrect / classTotal) * 100).toFixed(0)}%)`);
  console.log(`Topic routing:  ${topicCorrect}/${classTotal} (${((topicCorrect / classTotal) * 100).toFixed(0)}%)`);

  // 2. End-to-end search quality
  console.log("\n═══════ SEARCH QUALITY EVAL ═══════\n");

  const searchTests = [
    { query: "nativewind latest configuration for global.css", minResults: 3, expectDomain: "nativewind.dev" },
    { query: "React useEffect cleanup function", minResults: 5, expectDomain: "react.dev" },
    { query: "docker compose volumes bind mount syntax", minResults: 3, expectDomain: "docs.docker.com" },
    { query: "TypeScript discriminated unions pattern matching", minResults: 3 },
    { query: "python asyncio gather vs wait difference", minResults: 3 },
  ];

  for (const st of searchTests) {
    const start = Date.now();
    const result = await unifiedSearch({ query: st.query, depth: "normal" }, client, config);
    const elapsed = Date.now() - start;
    const hasExpectedDomain = st.expectDomain ? result.results.some(r => r.domain.includes(st.expectDomain!)) : true;

    const ok = result.results.length >= st.minResults && hasExpectedDomain;
    const icon = ok ? "✅" : "❌";
    const rrfTag = result.semantic_reranked ? "🧠RRF" : "keyword";
    console.log(`${icon} "${st.query.slice(0, 55)}" → ${result.results.length} results, ${result.engines_used.length} engines, ${elapsed}ms [${rrfTag}]`);
    if (st.expectDomain && !hasExpectedDomain) {
      console.log(`   ⚠️ Expected domain ${st.expectDomain} not found in results`);
    }
    if (result.results.length > 0) {
      console.log(`   Top: ${result.results[0].title.slice(0, 60)} (${result.results[0].domain})`);
    }
  }

  console.log("\n═══════ DONE ═══════");
}

runEval().catch((err) => {
  console.error("Fatal eval error:", err);
  process.exit(1);
});
