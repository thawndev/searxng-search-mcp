// ── Search Pipeline: intelligent expansion, cross-validation, result assembly ──

import type {
  SearchParams, SearchOutput, SearXNGRawResult, ServerConfig, QueryAnalysis,
} from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { rankResults } from "./ranking.js";
import { analyzeQuery, expandQueries } from "./query-analyzer.js";
import { searchCache } from "./cache.js";
import { LRUCache } from "./cache.js";

const SHORT_CACHE_TTL = 60_000;

export async function executeSearch(
  params: SearchParams,
  client: SearXNGClient,
  config: ServerConfig,
  preAnalysis?: QueryAnalysis,
): Promise<SearchOutput> {
  const startTime = Date.now();

  // Use pre-computed analysis if provided, otherwise analyze (async)
  const analysis = preAnalysis ?? await analyzeQuery(params.query);

  const maxResults = Math.min(params.max_results ?? 25, config.maxResultsPerQuery);

  // Check cache
  const cacheKey = LRUCache.makeKey({
    query: params.query,
    categories: params.categories,
    engines: params.engines,
    language: params.language,
    time_range: params.time_range,
    safesearch: params.safesearch,
    site_filters: params.site_filters,
    expand: params.expand_queries,
    pageno: params.pageno,
  });

  const cached = searchCache.get(cacheKey) as SearchOutput | undefined;
  if (cached) {
    return {
      ...cached,
      metadata: { ...cached.metadata, cached: true },
    };
  }

  // Determine queries
  const queries =
    params.expand_queries !== false
      ? expandQueries(params.query, analysis, params.site_filters)
      : [params.query];

  // No category restriction — let all engines participate
  const categories = params.categories || undefined;

  // Execute all queries concurrently
  const allRawResults: SearXNGRawResult[] = [];
  const allSuggestions: string[] = [];
  const allAnswers: string[] = [];
  const allCorrections: string[] = [];
  const allInfoboxes: SearchOutput["infoboxes"] = [];
  const enginesUsed = new Set<string>();
  const unresponsiveEngines: string[] = [];

  const searchPromises = queries.map((q) =>
    client
      .search({
        q,
        categories,
        engines: params.engines,
        language: params.language,
        time_range: params.time_range || (analysis.time_sensitive ? "month" : undefined),
        safesearch: params.safesearch,
        pageno: params.pageno,
      })
      .catch((err) => {
        console.error(`[searxng-mcp] Query failed: "${q}" — ${err}`);
        return null;
      }),
  );

  const responses = await Promise.all(searchPromises);

  for (const resp of responses) {
    if (!resp) continue;
    allRawResults.push(...resp.results);
    allSuggestions.push(...resp.suggestions);
    allAnswers.push(...resp.answers);
    allCorrections.push(...resp.corrections);

    for (const ib of resp.infoboxes) {
      allInfoboxes.push({
        title: ib.infobox,
        content: ib.content,
        urls: ib.urls,
        attributes: ib.attributes,
      });
    }

    for (const r of resp.results) {
      r.engines.forEach((e) => enginesUsed.add(e));
    }

    for (const [engine] of resp.unresponsive_engines) {
      unresponsiveEngines.push(engine);
    }
  }

  // If we got very few results and have paginated, try page 2
  if (allRawResults.length < 5 && !params.pageno) {
    const page2Promise = client
      .search({
        q: params.query,
        categories,
        engines: params.engines,
        language: params.language,
        time_range: params.time_range,
        safesearch: params.safesearch,
        pageno: 2,
      })
      .catch(() => null);

    const page2 = await page2Promise;
    if (page2) {
      allRawResults.push(...page2.results);
    }
  }

  // Rank with query-type-aware scoring
  const scored = rankResults(allRawResults, analysis.time_sensitive, analysis.type);

  // Rejection threshold
  const minScore = 0.03;
  const filtered = scored.filter((r) => r.relevance_score >= minScore);
  const results = filtered.slice(0, maxResults);

  const output: SearchOutput = {
    results,
    queries_used: queries,
    total_raw_results: allRawResults.length,
    engines_used: [...enginesUsed],
    suggestions: [...new Set(allSuggestions)],
    answers: [...new Set(allAnswers)],
    corrections: [...new Set(allCorrections)],
    infoboxes: allInfoboxes,
    unresponsive_engines: [...new Set(unresponsiveEngines)],
    metadata: {
      cached: false,
      search_duration_ms: Date.now() - startTime,
      result_count: results.length,
      query_type: analysis.type,
      time_sensitive: analysis.time_sensitive,
    },
  };

  const cacheTtl = analysis.time_sensitive ? SHORT_CACHE_TTL : config.cacheTtl;
  searchCache.set(cacheKey, output, cacheTtl);

  return output;
}
