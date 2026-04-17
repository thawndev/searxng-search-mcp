#!/usr/bin/env node
// ── SearXNG Search MCP Server — Entry Point ──

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { ServerConfig } from "./types.js";
import { createServer } from "./server.js";
import { warmupEmbeddings } from "./embeddings.js";

function loadConfig(): ServerConfig {
  const searxngUrl = process.env.SEARXNG_URL;
  if (!searxngUrl) {
    console.error(
      "[searxng-search-mcp] ERROR: SEARXNG_URL environment variable is required.\n" +
        "Set it to your SearXNG instance URL, e.g.: SEARXNG_URL=http://localhost:8080\n" +
        "See README.md for setup instructions.",
    );
    process.exit(1);
  }

  return {
    searxngUrl,
    apiKey: process.env.SEARXNG_API_KEY,
    cacheTtl: Number(process.env.CACHE_TTL_MS) || 300_000,
    requestTimeout: Number(process.env.REQUEST_TIMEOUT_MS) || 15_000,
    maxCacheSize: Number(process.env.MAX_CACHE_SIZE) || 500,
    maxConcurrentRequests: Number(process.env.MAX_CONCURRENT) || 8,
    maxResultsPerQuery: Number(process.env.MAX_RESULTS) || 50,
    maxFetchSize: Number(process.env.MAX_FETCH_SIZE) || 50_000,
    retry: {
      maxRetries: Number(process.env.MAX_RETRIES) || 2,
      baseDelayMs: Number(process.env.RETRY_BASE_DELAY_MS) || 500,
      maxDelayMs: Number(process.env.RETRY_MAX_DELAY_MS) || 5000,
    },
  };
}

async function main() {
  const config = loadConfig();
  const server = createServer(config);
  const transport = new StdioServerTransport();

  console.error(`[searxng-search-mcp] Starting...`);
  console.error(`[searxng-search-mcp] SearXNG: ${config.searxngUrl}`);
  console.error(`[searxng-search-mcp] Config: timeout=${config.requestTimeout}ms cache=${config.cacheTtl}ms concurrent=${config.maxConcurrentRequests} retries=${config.retry.maxRetries}`);
  console.error(`[searxng-search-mcp] Tools: search, ui_inspire, get_diagnostics`);

  await server.connect(transport);
  console.error(`[searxng-search-mcp] Connected and ready.`);

  // Warm up embedding model + archetype embeddings in background (non-blocking)
  warmupEmbeddings().then((ok) => {
    console.error(`[searxng-search-mcp] Embedding model: ${ok ? "ready — neural classification + RRF reranking active" : "unavailable, using syntax fallback"}`);
  });
}

main().catch((err) => {
  console.error("[searxng-search-mcp] Fatal error:", err);
  process.exit(1);
});
