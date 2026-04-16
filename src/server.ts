// ── MCP Server: tool registration and request handling ──

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { ServerConfig } from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { unifiedSearch, formatOutput } from "./unified-search.js";
import { searchCache, fetchCache, verifyCache } from "./cache.js";
import { getEmbeddingStats } from "./embeddings.js";

export function createServer(config: ServerConfig): McpServer {
  const server = new McpServer({
    name: "searxng-search",
    version: "4.0.0",
  });

  const client = new SearXNGClient(config);

  // ─── THE tool: search ───
  server.tool(
    "search",
    `Search the internet via SearXNG metasearch engine (10+ engines: Google, Bing, Brave, DuckDuckGo, Wikipedia, StackOverflow, GitHub, etc.). Queries are understood via neural embedding classification (bge-small-en-v1.5) and results are semantically reranked using Reciprocal Rank Fusion (RRF).

THIS IS YOUR GATEWAY TO ALL LIVE INFORMATION. Use it for:
- Any question requiring current/external information
- Looking up documentation, APIs, error messages, code examples
- Checking latest news, releases, announcements
- Verifying facts and claims
- Researching complex topics in depth
- Reading specific web pages by URL

HOW IT WORKS:
1. Your query is classified by neural embeddings (zero-shot, no regex patterns)
2. Search depth, topic, and freshness are auto-selected intelligently
3. Queries are expanded into variants for better recall
4. Results are gathered from 10+ engines, deduplicated, cross-validated
5. Results are semantically reranked using RRF (keyword + embedding + consensus)
6. For deep/code queries: top pages are auto-fetched and cleaned content included
7. Output is structured and token-efficient

MODES:
- Give a "query" to search the web
- Give a "url" to fetch and read a specific page
- Set "verify" to fact-check a claim with evidence
- Set depth="research" for multi-step iterative research

TIPS:
- Be specific: "React useEffect cleanup memory leak" > "react hooks"
- Use site filter: site=["github.com"] to restrict domains
- For breaking news: freshness="day" or "week"
- For claims: verify=true will search for supporting AND contradicting evidence
- depth="research" does multiple rounds of search with gap analysis
- For reading a known URL: just pass url="https://..."`,
    {
      query: z.string().optional().describe("The search query. Be specific for best results. Required unless 'url' is provided."),
      url: z.string().optional().describe("Fetch and read a specific URL directly. Use when you know the exact page you want. Mutually exclusive with 'query'."),
      depth: z.enum(["auto", "quick", "normal", "deep", "research"]).optional().describe(
        "Search depth. 'auto' (default) detects from query type. 'quick'=top results only. 'normal'=expanded queries. 'deep'=expanded + fetches page content. 'research'=multi-step iterative research with synthesis."
      ),
      topic: z.enum(["auto", "code", "news", "academic", "general"]).optional().describe(
        "Topic routing. 'auto' (default) detects from query. Routes to optimal engines and categories."
      ),
      freshness: z.enum(["any", "day", "week", "month", "year"]).optional().describe(
        "Filter by recency. Auto-detected for time-sensitive queries."
      ),
      include_content: z.boolean().optional().describe(
        "Fetch and include cleaned page content from top results. Auto-enabled for code/troubleshooting queries in deep mode."
      ),
      verify: z.boolean().optional().describe(
        "Fact-check mode: treats query as a claim, searches for supporting AND contradicting evidence, returns verdict."
      ),
      site: z.array(z.string()).optional().describe(
        "Restrict to specific domains: ['github.com', 'stackoverflow.com', 'docs.python.org']"
      ),
      max_sources: z.number().min(1).max(50).optional().describe(
        "Maximum number of sources to return. Default depends on depth (quick=8, normal=15, deep=20)."
      ),
      language: z.string().optional().describe("Language code: 'en', 'de', 'fr', 'es', 'zh', 'ja', etc."),
    },
    async (args) => {
      // Validate: need either query or url
      if (!args.query && !args.url) {
        return {
          content: [{ type: "text" as const, text: "Error: provide either 'query' (to search) or 'url' (to fetch a page)." }],
        };
      }
      if (args.query && args.url) {
        return {
          content: [{ type: "text" as const, text: "Error: provide either 'query' or 'url', not both." }],
        };
      }

      try {
        const result = await unifiedSearch(
          {
            query: args.query,
            url: args.url,
            depth: args.depth as any,
            topic: args.topic as any,
            freshness: args.freshness as any,
            include_content: args.include_content,
            verify: args.verify,
            site: args.site,
            max_sources: args.max_sources,
            language: args.language,
          },
          client,
          config,
        );

        const formatted = formatOutput(result);
        return {
          content: [{ type: "text" as const, text: formatted }],
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text" as const, text: `Search error: ${msg}` }],
        };
      }
    },
  );

  // ─── Diagnostics (debug-only, not for regular LLM use) ───
  server.tool(
    "get_diagnostics",
    "System health check. Returns SearXNG connectivity, cache stats, and config. Use only for debugging.",
    {},
    async () => {
      const health = await client.healthCheck();
      const embeddingStats = getEmbeddingStats();
      const stats = {
        healthy: health.healthy,
        instances: health.instances.map(i => ({
          url: i.url,
          reachable: i.reachable,
          latencyMs: i.latencyMs,
          queries: i.health.totalQueries,
          errors: i.health.totalErrors,
          avgLatencyMs: Math.round(i.health.avgLatencyMs),
        })),
        embeddings: embeddingStats,
        cache_search: searchCache.stats,
        cache_fetch: fetchCache.stats,
        cache_verify: verifyCache.stats,
      };
      return {
        content: [{ type: "text" as const, text: JSON.stringify(stats, null, 2) }],
      };
    },
  );

  return server;
}
