// ── Types for the SearXNG Search MCP server ──

// ─── SearXNG API Types ───

export interface SearXNGRawResult {
  url: string;
  title: string;
  content: string;
  engine: string;
  engines: string[];
  score: number;
  category: string;
  parsed_url?: string[];
  publishedDate?: string;
  img_src?: string;
  thumbnail_src?: string;
  template?: string;
  positions?: number[];
  // Image search fields
  resolution?: string;
  img_format?: string;
  source?: string;
}

export interface SearXNGInfobox {
  infobox: string;
  content: string;
  urls: Array<{ title: string; url: string }>;
  img_src?: string;
  attributes?: Array<{ label: string; value: string }>;
}

export interface SearXNGResponse {
  query: string;
  number_of_results: number;
  results: SearXNGRawResult[];
  answers: string[];
  corrections: string[];
  infoboxes: SearXNGInfobox[];
  suggestions: string[];
  unresponsive_engines: Array<[string, string]>;
}

// ─── Search Types ───

export interface SearchParams {
  query: string;
  categories?: string;
  engines?: string;
  language?: string;
  time_range?: "day" | "week" | "month" | "year";
  max_results?: number;
  safesearch?: 0 | 1 | 2;
  expand_queries?: boolean;
  site_filters?: string[];
  pageno?: number;
}

export interface ScoredResult {
  title: string;
  url: string;
  snippet: string;
  engines: string[];
  engine_count: number;
  relevance_score: number;
  semantic_score?: number; // Added by embedding reranker
  published_date?: string;
  domain: string;
  category: string;
}

export interface SearchOutput {
  results: ScoredResult[];
  queries_used: string[];
  total_raw_results: number;
  engines_used: string[];
  suggestions: string[];
  answers: string[];
  corrections: string[];
  infoboxes: Array<{
    title: string;
    content: string;
    urls: Array<{ title: string; url: string }>;
    attributes?: Array<{ label: string; value: string }>;
  }>;
  unresponsive_engines: string[];
  metadata: {
    cached: boolean;
    search_duration_ms: number;
    result_count: number;
    query_type: QueryType;
    time_sensitive: boolean;
  };
}

// ─── Multi-Search Types ───

export interface MultiSearchQuery {
  query: string;
  categories?: string;
  engines?: string;
  time_range?: "day" | "week" | "month" | "year";
  label?: string;
}

export interface MultiSearchOutput {
  searches: Array<{
    label: string;
    query: string;
    results: ScoredResult[];
    result_count: number;
  }>;
  combined_top_results: ScoredResult[];
  total_results: number;
  total_duration_ms: number;
}

// ─── Fetch Types ───

export interface FetchPageOutput {
  url: string;
  title: string;
  content: string;
  content_length: number;
  fetched_at: string;
  content_type: string;
  metadata: {
    description?: string;
    author?: string;
    published_date?: string;
    domain: string;
    headings?: string[];
    links_count?: number;
    word_count?: number;
    language?: string;
  };
  warning?: string;
}

// ─── Verify Types ───

export interface EvidenceItem {
  url: string;
  title: string;
  snippet: string;
  domain: string;
  supports_claim: "supports" | "contradicts" | "neutral";
  engines: string[];
  relevance_score: number;
}

export interface VerifyClaimOutput {
  claim: string;
  verdict: "supported" | "contradicted" | "insufficient_evidence" | "mixed";
  confidence: "high" | "medium" | "low";
  evidence: EvidenceItem[];
  queries_used: string[];
  summary: string;
  metadata: {
    supporting_count: number;
    contradicting_count: number;
    neutral_count: number;
    sources_checked: number;
    unique_domains: number;
  };
}

// ─── Deep Research Types ───

export interface ResearchStep {
  step: number;
  query: string;
  purpose: string;
  findings: ScoredResult[];
  gap_identified?: string;
}

export interface DeepResearchOutput {
  topic: string;
  steps: ResearchStep[];
  synthesis: string;
  key_findings: Array<{
    finding: string;
    sources: Array<{ url: string; title: string; domain: string }>;
    confidence: "high" | "medium" | "low";
  }>;
  open_questions: string[];
  total_sources: number;
  total_duration_ms: number;
}

// ─── Query Analysis Types ───

export type QueryType =
  | "factual"
  | "technical"
  | "news"
  | "academic"
  | "code"
  | "howto"
  | "comparison"
  | "definition"
  | "troubleshooting"
  | "general";

export interface QueryAnalysis {
  type: QueryType;
  confidence: number;        // top-2 gap (higher = more confident)
  secondary_type?: QueryType;
  time_sensitive: boolean;
  suggested_depth: "quick" | "normal" | "deep";
  suggested_topic: "code" | "news" | "academic" | "general";
  expansion_strategies: string[];
  key_entities: string[];
  neural: boolean;           // true if neural classification was used
}

// ─── Infrastructure Types ───

export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

export interface RetryConfig {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
}

export interface ServerConfig {
  searxngUrl: string;
  apiKey?: string;
  cacheTtl: number;
  requestTimeout: number;
  maxCacheSize: number;
  maxConcurrentRequests: number;
  maxResultsPerQuery: number;
  maxFetchSize: number;
  retry: RetryConfig;
}

// ─── Engine Info Types ───

export interface EngineInfo {
  name: string;
  shortcut: string;
  categories: string[];
  enabled: boolean;
  language_support: boolean;
  time_range_support: boolean;
  safesearch: boolean;
}
