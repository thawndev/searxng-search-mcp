// ── Unified Search: single intelligent entry point for all search operations ──
// Consolidates search, fetch, verify, and research into ONE tool.

import type {
  SearchOutput, ScoredResult, FetchPageOutput, VerifyClaimOutput,
  DeepResearchOutput, ServerConfig, QueryType,
} from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { executeSearch } from "./search.js";
import { fetchPage } from "./fetcher.js";
import { verifyClaim } from "./verifier.js";
import { deepResearch } from "./research.js";
import { analyzeQuery } from "./query-analyzer.js";
import { extractDomain } from "./ranking.js";
import { semanticRerank, warmupEmbeddings, isModelReady, getEmbeddingStats } from "./embeddings.js";

// ─── Types ───

export type SearchDepth = "auto" | "quick" | "normal" | "deep" | "research";
export type SearchTopic = "auto" | "code" | "news" | "academic" | "general";

export interface UnifiedSearchParams {
  query?: string;
  url?: string;
  depth?: SearchDepth;
  topic?: SearchTopic;
  freshness?: "any" | "day" | "week" | "month" | "year";
  include_content?: boolean;
  verify?: boolean;
  site?: string[];
  max_sources?: number;
  language?: string;
}

interface ResolvedPlan {
  mode: "search" | "fetch" | "verify" | "research";
  depth: Exclude<SearchDepth, "auto">;
  topic: Exclude<SearchTopic, "auto">;
  freshness?: "day" | "week" | "month" | "year";
  include_content: boolean;
  max_sources: number;
  fetch_budget: number; // max pages to auto-fetch content from
  reason: string;
}

export interface UnifiedSearchResult {
  plan: ResolvedPlan;
  // Search results
  results: ScoredResult[];
  total_sources: number;
  engines_used: string[];
  queries_used: string[];
  // Direct answers from SearXNG
  answers: string[];
  suggestions: string[];
  // Semantic embedding reranking applied
  semantic_reranked?: boolean;
  // Fetched content (when include_content is true)
  fetched_content?: Array<{
    source_id: string;
    url: string;
    title: string;
    domain: string;

    content: string;
    word_count: number;
  }>;
  // Verification (when verify mode)
  verification?: {
    verdict: string;
    confidence: string;
    supporting: number;
    contradicting: number;
    summary: string;
  };
  // Research synthesis (when research depth)
  research?: {
    synthesis: string;
    steps_taken: number;
    open_questions: string[];
  };
  // Timing
  duration_ms: number;
}

// ─── Plan Resolution ───

const DEPTH_FOR_TYPE: Record<QueryType, Exclude<SearchDepth, "auto">> = {
  troubleshooting: "deep",
  howto: "deep",
  code: "deep",
  comparison: "normal",
  academic: "normal",
  news: "quick",
  definition: "quick",
  factual: "normal",
  technical: "deep",
  general: "normal",
};

const TOPIC_FOR_TYPE: Record<QueryType, Exclude<SearchTopic, "auto">> = {
  code: "code",
  troubleshooting: "code",
  technical: "code",
  howto: "general",
  news: "news",
  academic: "academic",
  comparison: "general",
  definition: "general",
  factual: "general",
  general: "general",
};

// Types where auto-fetching page content is almost always helpful
const CONTENT_BENEFICIAL_TYPES = new Set<QueryType>([
  "code", "troubleshooting", "howto", "technical",
]);

const CATEGORIES_FOR_TOPIC: Record<Exclude<SearchTopic, "auto">, string | undefined> = {
  code: undefined,      // Don't restrict — general engines (Brave, Bing) have best code results
  news: "news",         // Only news actually benefits from category restriction
  academic: "science",
  general: undefined,   // No restriction
};

async function resolvePlan(params: UnifiedSearchParams): Promise<ResolvedPlan> {
  // Direct URL fetch mode
  if (params.url && !params.query) {
    return {
      mode: "fetch",
      depth: "deep",
      topic: "general",
      include_content: true,
      max_sources: 1,
      fetch_budget: 1,
      reason: "Direct URL fetch requested",
    };
  }

  const query = params.query || "";

  // Verify mode
  if (params.verify) {
    return {
      mode: "verify",
      depth: "deep",
      topic: "general",
      freshness: params.freshness === "any" ? undefined : params.freshness,
      include_content: false,
      max_sources: 20,
      fetch_budget: 0,
      reason: "Verification mode: searching for supporting and contradicting evidence",
    };
  }

  // Neural query analysis (async)
  const analysis = await analyzeQuery(query);
  const detectedType = analysis.type;

  // Use neural suggestions when available, with user overrides taking priority
  const topic: Exclude<SearchTopic, "auto"> =
    params.topic && params.topic !== "auto"
      ? params.topic
      : analysis.neural ? analysis.suggested_topic : TOPIC_FOR_TYPE[detectedType];

  let depth: Exclude<SearchDepth, "auto">;
  if (params.depth && params.depth !== "auto") {
    depth = params.depth;
  } else {
    depth = analysis.neural ? analysis.suggested_depth : DEPTH_FOR_TYPE[detectedType];
  }

  // Research mode
  if (depth === "research") {
    return {
      mode: "research",
      depth: "research",
      topic,
      freshness: params.freshness === "any" ? undefined : params.freshness,
      include_content: false,
      max_sources: params.max_sources || 30,
      fetch_budget: 0,
      reason: `Research mode: multi-step iterative search on "${query.slice(0, 50)}"`,
    };
  }

  // Resolve freshness
  let freshness = params.freshness === "any" ? undefined : params.freshness;
  if (!freshness && analysis.time_sensitive) {
    freshness = "month";
  }

  // Resolve include_content
  let includeContent: boolean;
  if (params.include_content !== undefined) {
    includeContent = params.include_content;
  } else {
    includeContent = depth === "deep" && CONTENT_BENEFICIAL_TYPES.has(detectedType);
  }

  // Resolve max_sources
  const maxSourcesDefault = depth === "quick" ? 8 : depth === "normal" ? 15 : 20;
  const maxSources = params.max_sources || maxSourcesDefault;

  // Fetch budget: how many pages to auto-fetch
  // Only fetch from high-confidence sources: official docs, multi-engine results
  const fetchBudget = includeContent ? Math.min(3, Math.ceil(maxSources / 5)) : 0;

  const reasons: string[] = [`Detected: ${detectedType}${analysis.neural ? " (neural)" : " (fallback)"}`];
  if (analysis.confidence > 0) reasons.push(`conf: ${analysis.confidence.toFixed(2)}`);
  if (analysis.time_sensitive) reasons.push("time-sensitive");
  if (includeContent) reasons.push(`will fetch top ${fetchBudget} pages`);

  return {
    mode: "search",
    depth,
    topic,
    freshness,
    include_content: includeContent,
    max_sources: maxSources,
    fetch_budget: fetchBudget,
    reason: reasons.join(" | "),
  };
}

// ─── Selective Content Fetching ───

// Domains worth auto-fetching (official docs, known high-quality)
const HIGH_VALUE_DOMAINS = new Set([
  // Official docs — major platforms
  "developer.mozilla.org", "docs.python.org", "doc.rust-lang.org",
  "nodejs.org", "go.dev", "learn.microsoft.com", "docs.oracle.com",
  "docs.github.com", "docs.docker.com", "kubernetes.io",
  // Frontend frameworks
  "react.dev", "vuejs.org", "angular.io", "svelte.dev",
  "nextjs.org", "nuxt.com", "astro.build", "remix.run",
  // CSS/styling
  "tailwindcss.com", "nativewind.dev", "styled-components.com",
  // Mobile
  "reactnative.dev", "docs.expo.dev", "flutter.dev",
  // Backend
  "docs.djangoproject.com", "flask.palletsprojects.com",
  "expressjs.com", "fastapi.tiangolo.com", "spring.io",
  // Cloud/infra
  "docs.aws.amazon.com", "cloud.google.com", "vercel.com/docs",
  // Community/reference
  "stackoverflow.com", "github.com", "dev.to",
  "wiki.archlinux.org", "en.wikipedia.org",
  "typescriptlang.org", "www.typescriptlang.org",
  "man7.org", "pkg.go.dev", "crates.io", "npmjs.com",
  // DB/tools
  "prisma.io", "orm.drizzle.team", "supabase.com/docs",
]);

function selectPagesToFetch(
  results: ScoredResult[],
  budget: number,
): ScoredResult[] {
  if (budget <= 0 || results.length === 0) return [];

  // Score each result for fetch-worthiness
  const scored = results.map((r) => {
    let fetchScore = r.relevance_score;
    // Boost official/authoritative domains
    if (HIGH_VALUE_DOMAINS.has(r.domain)) fetchScore += 0.3;
    // Boost multi-engine corroboration
    if (r.engine_count >= 2) fetchScore += 0.15;
    // Boost docs-like URLs
    if (r.url.includes("/docs/") || r.url.includes("/api/") || r.url.includes("/reference/")) {
      fetchScore += 0.1;
    }
    return { result: r, fetchScore };
  });

  // Sort by fetch-worthiness, take top N
  scored.sort((a, b) => b.fetchScore - a.fetchScore);

  // Diversify: don't fetch multiple pages from same domain
  const selected: ScoredResult[] = [];
  const seenDomains = new Set<string>();
  for (const { result } of scored) {
    if (selected.length >= budget) break;
    if (seenDomains.has(result.domain)) continue;
    seenDomains.add(result.domain);
    selected.push(result);
  }

  return selected;
}

// ─── Output Formatting ───

function truncateContent(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;
  // Try to truncate at a paragraph or sentence boundary
  const truncPoint = content.lastIndexOf("\n\n", maxChars);
  if (truncPoint > maxChars * 0.7) {
    return content.slice(0, truncPoint) + "\n\n[...truncated]";
  }
  const sentenceEnd = content.lastIndexOf(". ", maxChars);
  if (sentenceEnd > maxChars * 0.7) {
    return content.slice(0, sentenceEnd + 1) + " [...truncated]";
  }
  return content.slice(0, maxChars) + "... [truncated]";
}

export function formatOutput(result: UnifiedSearchResult): string {
  const lines: string[] = [];
  const p = result.plan;

  // Resolution header — shows the LLM what the tool decided
  const rrfTag = result.semantic_reranked ? " | 🧠 RRF-reranked" : "";
  lines.push(`**Search** | ${p.topic} | depth: ${p.depth} | ${result.total_sources} sources | ${result.engines_used.join(", ")}${rrfTag} | ${result.duration_ms}ms`);
  lines.push(`*Plan: ${p.reason}*`);

  // Direct answers (high priority — SearXNG sometimes has instant answers)
  if (result.answers.length > 0) {
    lines.push("");
    lines.push("### Direct Answer");
    for (const a of result.answers) {
      lines.push(a);
    }
  }

  // Verification results
  if (result.verification) {
    const v = result.verification;
    lines.push("");
    lines.push(`### Verification: **${v.verdict}** (${v.confidence} confidence)`);
    lines.push(`Supporting: ${v.supporting} | Contradicting: ${v.contradicting}`);
    lines.push(v.summary);
  }

  // Research synthesis
  if (result.research) {
    lines.push("");
    lines.push("### Research Synthesis");
    lines.push(`*${result.research.steps_taken} research steps, ${result.total_sources} sources*`);
    lines.push("");
    lines.push(result.research.synthesis);
    if (result.research.open_questions.length > 0) {
      lines.push("");
      lines.push("**Open questions:** " + result.research.open_questions.join(" | "));
    }
  }

  // Search results — compact format
  if (result.results.length > 0) {
    lines.push("");
    lines.push("### Results");
    for (let i = 0; i < result.results.length; i++) {
      const r = result.results[i];
      const id = `[${i + 1}]`;
      const engines = r.engine_count > 1 ? ` [${r.engines.join("+")}]` : "";
      const date = r.published_date ? ` (${r.published_date})` : "";
      lines.push(`${id} **${r.title}** — ${r.domain}${engines}${date}`);
      lines.push(`    ${r.url}`);
      if (r.snippet) {
        lines.push(`    ${r.snippet.slice(0, 200)}`);
      }
    }
  }

  // Fetched content — the high-value extracted text
  if (result.fetched_content && result.fetched_content.length > 0) {
    lines.push("");
    lines.push("### Fetched Content");
    for (const fc of result.fetched_content) {
      lines.push("");
      lines.push(`#### ${fc.source_id} ${fc.title} (${fc.domain}, ${fc.word_count} words)`);
      lines.push(fc.url);
      lines.push("");
      lines.push(fc.content);
    }
  }

  // Suggestions for follow-up
  if (result.suggestions.length > 0) {
    lines.push("");
    lines.push("**Follow-up searches:** " + result.suggestions.slice(0, 5).join(" | "));
  }

  return lines.join("\n");
}

// ─── Main Unified Search ───

export async function unifiedSearch(
  params: UnifiedSearchParams,
  client: SearXNGClient,
  config: ServerConfig,
): Promise<UnifiedSearchResult> {
  const startTime = Date.now();
  const plan = await resolvePlan(params);

  // ─── Mode: Direct URL Fetch ───
  if (plan.mode === "fetch" && params.url) {
    const fetched = await fetchPage(params.url, config, params.max_sources ? params.max_sources * 10000 : undefined);
    return {
      plan,
      results: [],
      total_sources: 1,
      engines_used: [],
      queries_used: [],
      answers: [],
      suggestions: [],
      fetched_content: [{
        source_id: "[1]",
        url: fetched.url,
        title: fetched.title,
        domain: fetched.metadata.domain,
        content: truncateContent(fetched.content, config.maxFetchSize),
        word_count: fetched.metadata.word_count || 0,
      }],
      duration_ms: Date.now() - startTime,
    };
  }

  const query = params.query!;

  // ─── Mode: Verify Claim ───
  if (plan.mode === "verify") {
    const verifyResult = await verifyClaim(query, undefined, client, config);
    return {
      plan,
      results: verifyResult.evidence.map((e, i) => ({
        title: e.title,
        url: e.url,
        snippet: e.snippet,
        engines: e.engines,
        engine_count: e.engines.length,
        relevance_score: e.relevance_score,
        domain: e.domain,
        category: "general",
      })),
      total_sources: verifyResult.metadata.sources_checked,
      engines_used: [],
      queries_used: verifyResult.queries_used,
      answers: [],
      suggestions: [],
      verification: {
        verdict: verifyResult.verdict,
        confidence: verifyResult.confidence,
        supporting: verifyResult.metadata.supporting_count,
        contradicting: verifyResult.metadata.contradicting_count,
        summary: verifyResult.summary,
      },
      duration_ms: Date.now() - startTime,
    };
  }

  // ─── Mode: Deep Research ───
  if (plan.mode === "research") {
    const researchResult = await deepResearch(query, 3, undefined, client, config);
    const allFindings = researchResult.steps.flatMap((s) => s.findings);
    return {
      plan,
      results: allFindings.slice(0, plan.max_sources),
      total_sources: researchResult.total_sources,
      engines_used: [],
      queries_used: researchResult.steps.map((s) => s.query),
      answers: [],
      suggestions: [],
      research: {
        synthesis: researchResult.synthesis,
        steps_taken: researchResult.steps.length,
        open_questions: researchResult.open_questions,
      },
      duration_ms: Date.now() - startTime,
    };
  }

  // ─── Mode: Search (quick / normal / deep) ───
  const categories = plan.topic !== "general" ? CATEGORIES_FOR_TOPIC[plan.topic] : undefined;
  const expandQueries = plan.depth !== "quick";

  const searchOutput = await executeSearch(
    {
      query,
      categories: categories || undefined, // Never pass empty string
      language: params.language,
      time_range: plan.freshness,
      max_results: plan.max_sources,
      expand_queries: expandQueries,
      site_filters: params.site,
    },
    client,
    config,
  );

  // ─── Semantic Reranking via Embeddings ───
  // Uses RRF (Reciprocal Rank Fusion) to combine keyword, semantic, and consensus signals
  let rerankedResults = searchOutput.results;
  let usedSemanticRerank = false;

  if (rerankedResults.length > 0) {
    try {
      rerankedResults = await semanticRerank(query, rerankedResults);
      usedSemanticRerank = true;
    } catch {
      // Graceful degradation — keep keyword-ranked results
    }
  }

  // ─── Fallback: if news category returned 0 results, retry without category ───
  if (rerankedResults.length === 0 && categories) {
    const fallbackOutput = await executeSearch(
      {
        query,
        language: params.language,
        time_range: plan.freshness,
        max_results: plan.max_sources,
        expand_queries: expandQueries,
        site_filters: params.site,
      },
      client,
      config,
    );
    rerankedResults = fallbackOutput.results;

    if (rerankedResults.length > 0) {
      try {
        rerankedResults = await semanticRerank(query, rerankedResults);
        usedSemanticRerank = true;
      } catch {
        // Graceful degradation
      }
    }

    // Merge engines and answers from fallback
    for (const e of fallbackOutput.engines_used) {
      if (!searchOutput.engines_used.includes(e)) {
        searchOutput.engines_used.push(e);
      }
    }
    searchOutput.answers.push(...fallbackOutput.answers);
    searchOutput.suggestions.push(...fallbackOutput.suggestions);
    searchOutput.queries_used.push(...fallbackOutput.queries_used);
  }

  const output: UnifiedSearchResult = {
    plan,
    results: rerankedResults,
    total_sources: rerankedResults.length,
    engines_used: searchOutput.engines_used,
    queries_used: [...new Set(searchOutput.queries_used)],
    answers: [...new Set(searchOutput.answers)],
    suggestions: [...new Set(searchOutput.suggestions)],
    semantic_reranked: usedSemanticRerank,
    duration_ms: Date.now() - startTime,
  };

  // ─── Auto-fetch content for deep mode ───
  if (plan.include_content && plan.fetch_budget > 0 && searchOutput.results.length > 0) {
    const toFetch = selectPagesToFetch(searchOutput.results, plan.fetch_budget);

    const fetchPromises = toFetch.map(async (r, idx) => {
      try {
        const maxLen = Math.floor(config.maxFetchSize / plan.fetch_budget);
        const fetched = await fetchPage(r.url, config, maxLen);
        return {
          source_id: `[${searchOutput.results.indexOf(r) + 1}]`,
          url: r.url,
          title: fetched.title || r.title,
          domain: r.domain,
          content: truncateContent(fetched.content, maxLen),
          word_count: fetched.metadata.word_count || 0,
        };
      } catch {
        return null;
      }
    });

    const fetched = (await Promise.all(fetchPromises)).filter(Boolean) as NonNullable<Awaited<typeof fetchPromises[0]>>[];
    if (fetched.length > 0) {
      output.fetched_content = fetched;
    }
    output.duration_ms = Date.now() - startTime;
  }

  return output;
}
