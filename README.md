# SearXNG Search MCP

**Neural-powered web search for AI development workflows.**

A single intelligent `search` tool that gives your AI agent verified, grounded internet access. Built on [SearXNG](https://github.com/searxng/searxng) metasearch (50+ engines), with zero-shot neural query classification and semantic RRF reranking via [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings.

Designed for [GitHub Copilot CLI](https://docs.github.com/en/copilot), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), and any MCP-compatible agent.

## Features

- **One tool, all capabilities** — search, fetch pages, verify claims, deep research
- **Neural query understanding** — zero-shot embedding classification, no hardcoded patterns
- **Semantic reranking** — Reciprocal Rank Fusion combines keyword, embedding, and consensus signals
- **Cross-validated results** — 50+ engines with deduplication and multi-engine corroboration
- **Source-attributed output** — every result traced to origin, structured for token efficiency
- **Auto content fetching** — top pages fetched, cleaned, and included for deep queries
- **Multi-instance failover** — hedged requests across SearXNG instances with health tracking
- **Built-in caching** — LRU cache with TTL for repeat queries

## Quick Start

### 1. Start SearXNG

```bash
# Copy template config
mkdir -p searxng-data
cp searxng-config/settings.yml searxng-data/settings.yml

# Start container
docker compose up -d
```

### 2. Install & Build

```bash
npm install && npm run build
```

### 3. Configure MCP

**GitHub Copilot CLI** — add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "searxng-search": {
      "command": "node",
      "args": ["/absolute/path/to/searxng-search-mcp/dist/index.js"],
      "env": {
        "SEARXNG_URL": "http://localhost:8080"
      }
    }
  }
}
```

**Claude Code** — add to `.claude/mcp.json` or `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "searxng-search": {
      "command": "node",
      "args": ["/absolute/path/to/searxng-search-mcp/dist/index.js"],
      "env": {
        "SEARXNG_URL": "http://localhost:8080"
      }
    }
  }
}
```

### 4. Restart Your CLI Session

MCP tools are loaded at session start. After adding config, restart your terminal agent.

## The `search` Tool

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Search query. Required unless `url` is provided. |
| `url` | string | — | Fetch a specific URL directly. Mutually exclusive with `query`. |
| `depth` | `auto\|quick\|normal\|deep\|research` | `auto` | Search depth. Auto-detects from query. |
| `topic` | `auto\|code\|news\|academic\|general` | `auto` | Topic routing. Auto-detects from query. |
| `freshness` | `any\|day\|week\|month\|year` | auto | Filter by recency. Auto-detected for time-sensitive queries. |
| `include_content` | boolean | auto | Fetch cleaned page content from top results. Auto-enabled for code/troubleshooting in deep mode. |
| `verify` | boolean | false | Fact-check mode: searches for supporting AND contradicting evidence. |
| `site` | string[] | — | Restrict to domains: `["github.com", "docs.python.org"]` |
| `max_sources` | number (1-50) | auto | Max sources. Default: quick=8, normal=15, deep=20. |
| `language` | string | — | Language code: `en`, `de`, `fr`, etc. |

### Depth Modes

- **`quick`** — Single query, top results only (~1s)
- **`normal`** — Expanded queries for better recall (~2-3s)
- **`deep`** — Expanded queries + auto-fetches page content from top results (~5-8s)
- **`research`** — Multi-step iterative search: search → analyze gaps → refine → synthesize (~10-15s)

### Intelligent Auto-Detection

When `depth` and `topic` are `auto` (default), the neural classifier analyzes your query and routes intelligently:

- Error messages → deep code search with content fetching
- Documentation lookups → deep search with code topic
- Breaking news queries → quick search with freshness filter
- Academic queries → normal search with science category
- Simple definitions → quick search
- Complex comparisons → normal search with structured expansion

The classifier uses zero-shot cosine similarity against natural language archetype descriptions — no regex patterns or keyword matching. This means it generalizes to novel query phrasings without maintenance.

### Output Format

Structured markdown optimized for LLM consumption:

```
**Search** | code | depth: deep | 20 sources | bing, google, mojeek | 🧠 RRF-reranked | 3200ms
*Plan: Detected: code (neural) | conf: 0.15 | will fetch top 3 pages*

### Direct Answer
(instant answers from SearXNG, if available)

### Results
[1] **Title** — domain.com [google+bing] (2024-01-15)
    https://url...
    Snippet text...

### Fetched Content
#### [1] Page Title (domain.com, 1500 words)
(cleaned, readable page content)

**Follow-up searches:** suggestion1 | suggestion2
```

## How It Works

### Neural Query Classification

At startup, 45+ natural language archetype descriptions are embedded using bge-small-en-v1.5:

```
code:     "programming code function API library package npm pip cargo"
news:     "breaking news announcement today politics economy business"
academic: "research paper scientific study journal peer reviewed"
```

At query time, the query is embedded and classified via **max cosine similarity** against archetypes. The **top-2 gap** serves as a confidence signal. Separate classifiers handle type, freshness, depth, and topic independently — so a code query containing temporal words like "latest" is correctly routed to code, not news, because the freshness classifier requires a clear margin over the timeless baseline.

When the embedding model is still loading (~1.6s), a minimal syntax fallback handles queries.

### Semantic RRF Reranking

Results are reranked using Reciprocal Rank Fusion, combining three signals:

1. **Keyword rank** — SearXNG's native relevance score
2. **Semantic rank** — cosine similarity of query vs title/snippet embeddings (0.7/0.3 weighted)
3. **Consensus rank** — number of engines returning the same result

Formula: `score = 1/(k + keywordRank) + 1/(k + semanticRank) + 1/(k + consensusRank)` where k=60.

### Adaptive Ranking

Beyond RRF, results are scored with query-type-aware weights:

- **Code queries** — higher weight on snippet quality and relevance
- **News queries** — higher weight on freshness
- **Academic queries** — higher weight on domain reputation
- **Factual queries** — higher weight on corroboration

Domain reputation tiers (tier-1: official docs, academic publishers; tier-2: quality tech blogs; penalty: content farms) provide a baseline signal.

## Engine Resilience

### Multi-Instance Hedged Failover

Set `SEARXNG_URL` to comma-separated URLs for automatic failover:

```bash
SEARXNG_URL=http://localhost:8080,http://localhost:8081
```

- Health tracking per instance (latency, error rate, suspension)
- Hedged requests: if primary is slow (>2s) or returns few results, secondary fires automatically
- Instance auto-suspension after 5 consecutive errors with exponential backoff

### Optimized SearXNG Configuration

The included `searxng-config/settings.yml` template configures:

- Reduced engine suspension times (CAPTCHA: 5 min vs default 1 hour)
- Higher timeouts (10s default, 15s max)
- 50+ engines across categories (general, code, news, academic)
- Larger connection pool (100 connections, HTTP/2 enabled)
- Rate limiter disabled for local use

### Optional Tor Proxy

For IP rotation to avoid rate limiting:

1. Uncomment the `tor-proxy` service in `docker-compose.yml`
2. Add proxy config to `searxng-data/settings.yml`:
   ```yaml
   outgoing:
     proxies:
       all://:
         - socks5h://tor-proxy:9050
     using_tor_proxy: true
     extra_proxy_timeout: 10
   ```
3. `docker compose up -d`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SEARXNG_URL` | *required* | SearXNG URL(s). Comma-separated for multi-instance. |
| `SEARXNG_API_KEY` | — | Optional API key |
| `CACHE_TTL_MS` | 300000 | Cache TTL (5 min) |
| `REQUEST_TIMEOUT_MS` | 15000 | HTTP timeout |
| `MAX_CACHE_SIZE` | 500 | Max cache entries |
| `MAX_CONCURRENT` | 8 | Max parallel requests |
| `MAX_RESULTS` | 50 | Max results per query |
| `MAX_FETCH_SIZE` | 50000 | Max fetched page size (chars) |
| `MAX_RETRIES` | 2 | Retry attempts |
| `RETRY_BASE_DELAY_MS` | 500 | Base retry delay |
| `RETRY_MAX_DELAY_MS` | 5000 | Max retry delay |

## Troubleshooting

### MCP not showing in Copilot CLI

The correct config path is `~/.copilot/mcp-config.json`. After adding it, restart the CLI session. Paths in `args` must be absolute.

### Engines getting suspended

Check the SearXNG UI at http://localhost:8080 → Preferences → Engines tab. Quick fixes:
1. Restart SearXNG: `docker compose restart`
2. Add a second instance for failover
3. Enable Tor proxy for IP rotation

### No results returned

1. Verify SearXNG is running: `curl http://localhost:8080/search?q=test&format=json`
2. Check Docker logs: `docker compose logs -f searxng`
3. Some engines need time to initialize on first start

### Slow first search

The embedding model loads on first use (~1.6s). Subsequent queries use cached embeddings (<5ms classification, <1ms for cached searches).

## Architecture

```
┌─────────────────────────────────────────────┐
│              MCP Client (LLM)               │
│         "search" tool (unified)             │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│            Unified Search Engine             │
│  ┌───────────────────────────────────────┐  │
│  │  Neural Query Intelligence            │  │
│  │  bge-small-en-v1.5 embeddings         │  │
│  │  ┌─────────────┐ ┌─────────────────┐ │  │
│  │  │ Zero-shot   │ │ Semantic RRF    │ │  │
│  │  │ Classifier  │ │ Reranker        │ │  │
│  │  │ (archetype  │ │ (keyword+embed  │ │  │
│  │  │  matching)  │ │  +consensus)    │ │  │
│  │  └─────────────┘ └─────────────────┘ │  │
│  └───────────────────────────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │  Search  │ │ Fetcher  │ │  Verifier   │ │
│  │ Pipeline │ │ (HTML→   │ │ (evidence   │ │
│  │ (expand, │ │  clean   │ │  classify)  │ │
│  │  rank)   │ │  text)   │ │             │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Research │ │  Cache   │ │  Ranking    │ │
│  │ (multi-  │ │ (LRU +   │ │ (adaptive   │ │
│  │  step)   │ │  TTL)    │ │  weights)   │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│    SearXNG Client (multi-instance, hedge)   │
└────────┬─────────────────────────┬──────────┘
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │ SearXNG │              │ SearXNG │
    │ Primary │              │Secondary│
    │ :8080   │              │ :8081   │
    └─────────┘              └─────────┘
```

## Development

```bash
# Run tests (requires SearXNG running on localhost:8080)
npm run build && node dist/test.js

# Run eval harness (classification accuracy + search quality)
node dist/benchmark.js

# Watch mode
npm run dev
```

## License

MIT
