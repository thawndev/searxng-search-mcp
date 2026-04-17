// ── MCP Server: tool registration and request handling ──

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { ServerConfig } from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { unifiedSearch, formatOutput } from "./unified-search.js";
import { searchCache, fetchCache, verifyCache } from "./cache.js";
import { getEmbeddingStats } from "./embeddings.js";
import { uiInspire, formatUIInspireOutput, imageCache } from "./ui-inspire.js";

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
        cache_images: imageCache.stats,
      };
      return {
        content: [{ type: "text" as const, text: JSON.stringify(stats, null, 2) }],
      };
    },
  );

  // ─── UI Inspiration Engine ───
  server.tool(
    "ui_inspire",
    `Search for UI/UX design inspiration using SearXNG image search across Dribbble, Behance, Pinterest, Unsplash, and 10+ image engines. Returns curated design references with optional image thumbnails that vision-capable LLMs can directly analyze.

USE THIS FOR:
- Finding UI design inspiration for screens, layouts, or components
- Exploring design patterns (login flows, dashboards, onboarding, settings)
- Discovering visual styles (glassmorphism, neomorphism, minimal, material)
- Getting real design references before building UI components
- Combining visual inspiration with code examples for implementation

HOW IT WORKS:
1. Your design intent is expanded into optimized image search queries
2. Images are gathered from design-focused engines (Dribbble, Behance, Pinterest, etc.)
3. Results are ranked by design relevance: domain reputation + semantic similarity + resolution quality
4. Top images are fetched as thumbnails and returned as viewable image content
5. Optionally includes code references for the requested framework

MODES:
- "thumbnails" (default): Returns 2-3 thumbnail images + metadata — best for quick inspiration
- "links_only": Returns only metadata and URLs — no image downloads, fastest
- "inspect": Returns 1 high-resolution image for detailed analysis

TIPS:
- Be specific: "mobile banking app dashboard dark theme" > "app design"
- Use 'platform' to filter: "mobile", "web", "tablet", "desktop"
- Use 'style' for aesthetic direction: "glassmorphism", "minimal", "brutalist"
- Use 'components' to focus: ["navigation bar", "card grid", "hero section"]
- Add 'framework' to get code references alongside: "react-native tailwindcss"`,
    {
      query: z.string().describe("Design intent or UI search query. Be specific about the screen type, style, and purpose."),
      style: z.string().optional().describe("Visual style filter: 'minimal', 'glassmorphism', 'neomorphism', 'material', 'flat', 'brutalist', 'skeuomorphic', etc."),
      platform: z.enum(["mobile", "web", "tablet", "desktop"]).optional().describe("Target platform to filter results."),
      components: z.array(z.string()).optional().describe("Specific UI components to focus on: ['login form', 'navigation', 'cards', 'hero section']"),
      max_images: z.number().min(1).max(6).optional().describe("Number of images to return (default: 3). Keep low for token efficiency."),
      mode: z.enum(["thumbnails", "links_only", "inspect"]).optional().describe(
        "Output mode. 'thumbnails' (default): small preview images. 'links_only': metadata only, no downloads. 'inspect': 1 high-res image for detailed analysis."
      ),
      framework: z.string().optional().describe("If provided, also searches for code examples in this framework (e.g., 'react-native tailwindcss', 'nextjs', 'flutter')."),
      safesearch: z.number().min(0).max(2).optional().describe("Safe search level: 0=off, 1=moderate (default), 2=strict."),
    },
    async (args) => {
      try {
        const result = await uiInspire(
          {
            query: args.query,
            style: args.style,
            platform: args.platform as any,
            components: args.components,
            max_images: args.max_images,
            mode: args.mode as any,
            framework: args.framework,
            safesearch: args.safesearch as any,
          },
          client,
          config,
        );

        const formatted = formatUIInspireOutput(result);
        return {
          content: formatted.map(block => {
            if (block.type === "image") {
              return { type: "image" as const, data: block.data!, mimeType: block.mimeType! };
            }
            return { type: "text" as const, text: block.text! };
          }),
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text" as const, text: `UI inspiration search error: ${msg}` }],
        };
      }
    },
  );

  return server;
}
