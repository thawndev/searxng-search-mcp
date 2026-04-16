// ── Source Ranking and Scoring — Multi-stage with adaptive weights ──

import type { SearXNGRawResult, ScoredResult, QueryType } from "./types.js";

// Domain reputation tiers — comprehensive and extensible
const TIER_1_DOMAINS = new Set([
  // Reference
  "wikipedia.org", "britannica.com", "merriam-webster.com",
  // Developer / Programming
  "github.com", "gitlab.com", "stackoverflow.com", "stackexchange.com",
  "developer.mozilla.org", "docs.python.org", "docs.rs", "pkg.go.dev",
  "learn.microsoft.com", "docs.microsoft.com", "cloud.google.com",
  "aws.amazon.com", "docs.aws.amazon.com", "nodejs.org",
  "typescriptlang.org", "rust-lang.org", "go.dev", "kotlinlang.org",
  "docs.oracle.com", "docs.spring.io", "docs.djangoproject.com",
  "reactjs.org", "react.dev", "vuejs.org", "angular.io", "svelte.dev",
  "npmjs.com", "pypi.org", "crates.io", "rubygems.org", "packagist.org",
  // Standards
  "rfc-editor.org", "w3.org", "ietf.org", "ecma-international.org",
  // Academic
  "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
  "pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "semanticscholar.org",
  "researchgate.net", "jstor.org", "springer.com",
  // News (established)
  "nytimes.com", "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
  "washingtonpost.com", "theguardian.com", "economist.com",
  // Official
  "who.int", "un.org", "europa.eu", "gov.uk",
]);

const TIER_2_DOMAINS = new Set([
  "medium.com", "dev.to", "hashnode.com", "hackernoon.com",
  "arstechnica.com", "theverge.com", "wired.com", "techcrunch.com",
  "digitalocean.com", "freecodecamp.org", "geeksforgeeks.org",
  "towardsdatascience.com", "huggingface.co", "openai.com",
  "anthropic.com", "blog.google", "engineering.fb.com",
  "netflixtechblog.com", "uber.com", "stripe.com",
  "martinfowler.com", "kentcdodds.com", "joshwcomeau.com",
  "baeldung.com", "tutorialspoint.com", "w3schools.com",
  "cnbc.com", "bloomberg.com", "ft.com", "bbc.com",
]);

const LOW_QUALITY_DOMAINS = new Set([
  "pinterest.com", "quora.com", "answers.com",
  "ehow.com", "wikihow.com", "about.com",
]);

export function extractDomain(url: string): string {
  try {
    const hostname = new URL(url).hostname.replace(/^www\./, "");
    const parts = hostname.split(".");
    if (parts.length <= 2) return hostname;
    const sld = parts[parts.length - 2];
    if (["co", "com", "org", "net", "gov", "edu", "ac"].includes(sld)) {
      return parts.slice(-3).join(".");
    }
    return parts.slice(-2).join(".");
  } catch {
    return url;
  }
}

function domainReputation(domain: string): number {
  if (LOW_QUALITY_DOMAINS.has(domain)) return 0.15;
  if (TIER_1_DOMAINS.has(domain)) return 1.0;
  if (TIER_2_DOMAINS.has(domain)) return 0.7;
  for (const d of TIER_1_DOMAINS) {
    if (domain.endsWith(`.${d}`)) return 0.9;
  }
  for (const d of TIER_2_DOMAINS) {
    if (domain.endsWith(`.${d}`)) return 0.6;
  }
  // .edu and .gov get a boost
  if (domain.endsWith(".edu") || domain.endsWith(".gov")) return 0.85;
  return 0.4;
}

function freshnessScore(publishedDate?: string): number {
  if (!publishedDate) return 0.5;
  try {
    const pubDate = new Date(publishedDate);
    if (isNaN(pubDate.getTime())) return 0.5;
    const ageDays = (Date.now() - pubDate.getTime()) / (1000 * 60 * 60 * 24);
    if (ageDays < 0) return 0.5; // Future date = suspicious
    if (ageDays < 1) return 1.0;
    if (ageDays < 7) return 0.95;
    if (ageDays < 30) return 0.85;
    if (ageDays < 90) return 0.7;
    if (ageDays < 365) return 0.5;
    if (ageDays < 730) return 0.35;
    return 0.2;
  } catch {
    return 0.5;
  }
}

/** Assess snippet quality: length, information density, presence of data */
function snippetQuality(content: string): number {
  if (!content) return 0;
  const len = content.length;
  let score = Math.min(len / 300, 1.0) * 0.5;
  // Bonus for containing numbers/data
  if (/\d+/.test(content)) score += 0.1;
  // Bonus for containing code-like content
  if (/[`{}()=<>]/.test(content)) score += 0.1;
  // Bonus for sentences (well-formed content)
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 10);
  if (sentences.length >= 2) score += 0.15;
  // Penalty for boilerplate
  if (/\b(click here|subscribe|sign up|cookie|privacy policy)\b/i.test(content)) score -= 0.2;
  return Math.max(0, Math.min(score, 1.0));
}

function deduplicateResults(
  results: SearXNGRawResult[],
): Map<string, SearXNGRawResult & { all_engines: string[] }> {
  const seen = new Map<string, SearXNGRawResult & { all_engines: string[] }>();

  for (const r of results) {
    let canonical: string;
    try {
      const u = new URL(r.url);
      u.hash = "";
      for (const p of ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "ref", "source", "fbclid", "gclid"]) {
        u.searchParams.delete(p);
      }
      canonical = u.toString().replace(/\/+$/, "");
    } catch {
      canonical = r.url;
    }

    const existing = seen.get(canonical);
    if (existing) {
      const enginesSet = new Set([...existing.all_engines, ...r.engines, r.engine]);
      existing.all_engines = [...enginesSet];
      if (r.score > existing.score) {
        existing.title = r.title || existing.title;
        existing.content = r.content || existing.content;
        existing.score = r.score;
      }
      if (r.publishedDate && !existing.publishedDate) {
        existing.publishedDate = r.publishedDate;
      }
    } else {
      seen.set(canonical, {
        ...r,
        all_engines: [...new Set([...r.engines, r.engine])],
      });
    }
  }

  return seen;
}

/** Adaptive ranking weights based on query type */
function getWeights(queryType: QueryType, timeSensitive: boolean) {
  const base = {
    relevance: 0.35,
    corroboration: 0.2,
    reputation: 0.15,
    freshness: timeSensitive ? 0.25 : 0.05,
    snippetQuality: 0.1,
  };

  switch (queryType) {
    case "news":
      return { ...base, freshness: 0.35, relevance: 0.25, reputation: 0.15 };
    case "academic":
      return { ...base, reputation: 0.3, relevance: 0.3, freshness: 0.05, snippetQuality: 0.15 };
    case "code":
    case "troubleshooting":
      return { ...base, relevance: 0.35, snippetQuality: 0.2, corroboration: 0.15, reputation: 0.15 };
    case "definition":
    case "factual":
      return { ...base, reputation: 0.25, relevance: 0.3, corroboration: 0.2 };
    default:
      return base;
  }
}

/** Score and rank search results with adaptive weights */
export function rankResults(
  rawResults: SearXNGRawResult[],
  timeSensitive: boolean = false,
  queryType: QueryType = "general",
): ScoredResult[] {
  const deduped = deduplicateResults(rawResults);
  const weights = getWeights(queryType, timeSensitive);
  const scored: ScoredResult[] = [];

  for (const [, result] of deduped) {
    const domain = extractDomain(result.url);
    const engineCount = result.all_engines.length;

    const relevanceComponent = Math.min(result.score / 8, 1.0) * weights.relevance;
    const corroborationComponent = Math.min((engineCount - 1) * 0.15, 1.0) * weights.corroboration;
    const reputationComponent = domainReputation(domain) * weights.reputation;
    const freshnessComponent = freshnessScore(result.publishedDate) * weights.freshness;
    const qualityComponent = snippetQuality(result.content) * weights.snippetQuality;

    const totalScore =
      relevanceComponent + corroborationComponent + reputationComponent +
      freshnessComponent + qualityComponent;

    scored.push({
      title: result.title || "(No title)",
      url: result.url,
      snippet: result.content || "",
      engines: result.all_engines,
      engine_count: engineCount,
      relevance_score: Math.round(totalScore * 1000) / 1000,
      published_date: result.publishedDate,
      domain,
      category: result.category || "general",
    });
  }

  scored.sort((a, b) => b.relevance_score - a.relevance_score);
  return scored;
}
