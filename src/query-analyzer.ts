// ── Query Intelligence: neural classification with syntax fallback ──
// Uses embedding-based zero-shot classification (no hardcoded regex patterns for type detection).
// Minimal syntax heuristics only for structural detection (URLs, file extensions, code symbols).

import type { QueryAnalysis, QueryType } from "./types.js";
import { classifyQuery, isModelReady } from "./embeddings.js";

// ─── Syntax Heuristics (structural detection for URLs, file extensions, code symbols) ───

function detectUrl(query: string): boolean {
  return /^https?:\/\//i.test(query.trim());
}

function hasFileExtension(query: string): boolean {
  return /\b\w+\.(css|js|ts|tsx|jsx|json|yml|yaml|toml|py|rs|go|java|rb|php|sh|md|html|xml|env|config|lock|sql|graphql)\b/i.test(query);
}

function hasCodeSymbols(query: string): boolean {
  return /[`{}<>\[\]()=;]|\/\/|#!|->|=>|\.\.\.|\$\{/.test(query);
}

function hasVersionString(query: string): boolean {
  return /\bv?\d+\.\d+(?:\.\d+)?\b/.test(query);
}

// ─── Fallback Classification (used when model not loaded yet) ───

function fallbackClassify(query: string): QueryAnalysis {
  let type: QueryType = "general";
  let depth: "quick" | "normal" | "deep" = "normal";
  let topic: "code" | "news" | "academic" | "general" = "general";

  if (hasFileExtension(query) || hasCodeSymbols(query)) {
    type = "code";
    depth = "deep";
    topic = "code";
  } else if (/^(what|who) (is|are|was|were) /i.test(query)) {
    type = "definition";
    depth = "quick";
  } else if (/^how (to|do|can|should)/i.test(query)) {
    type = "howto";
    depth = "deep";
  }

  return {
    type,
    confidence: 0,
    time_sensitive: false,
    suggested_depth: depth,
    suggested_topic: topic,
    expansion_strategies: determineExpansionStrategies(query, type),
    key_entities: extractEntities(query),
    neural: false,
  };
}

// ─── Main Analysis Function ───

/** Analyze query using neural classification (embedding similarity) with syntax fallback. */
export async function analyzeQuery(query: string): Promise<QueryAnalysis> {
  // Try neural classification if model is ready (non-blocking check)
  if (isModelReady()) {
    const neural = await classifyQuery(query);
    if (neural) {
      return {
        type: neural.type,
        confidence: neural.confidence,
        secondary_type: neural.secondaryType,
        time_sensitive: neural.timeSensitive,
        suggested_depth: neural.suggestedDepth,
        suggested_topic: neural.suggestedTopic,
        expansion_strategies: determineExpansionStrategies(query, neural.type),
        key_entities: extractEntities(query),
        neural: true,
      };
    }
  }

  // Fallback: minimal syntax heuristics
  return fallbackClassify(query);
}

// ─── Entity Extraction (structural — regex is correct here) ───

function extractEntities(query: string): string[] {
  const entities: string[] = [];

  // Quoted phrases
  const quoted = query.match(/"([^"]+)"/g);
  if (quoted) entities.push(...quoted.map((q) => q.replace(/"/g, "")));

  // Capitalized words (likely proper nouns)
  const caps = query.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g);
  if (caps) entities.push(...caps);

  // Technical terms (camelCase, snake_case, kebab-case, or with dots)
  const technical = query.match(/\b[a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+(?:[-_.][a-z]+)+\b/g);
  if (technical) entities.push(...technical);

  // Version numbers
  const versions = query.match(/v?\d+\.\d+(?:\.\d+)?/g);
  if (versions) entities.push(...versions);

  return [...new Set(entities)].slice(0, 10);
}

// ─── Query Expansion ───

function determineExpansionStrategies(query: string, type: QueryType): string[] {
  const strategies: string[] = ["original"];
  const words = query.trim().split(/\s+/);

  if (type === "code" || type === "troubleshooting" || type === "technical") {
    strategies.push("add_context");
    if (query.length < 80) strategies.push("add_documentation_suffix");
  } else if (type === "howto") {
    strategies.push("add_context");
  } else if (type === "academic") {
    strategies.push("scholarly_rephrase");
  } else if (type === "definition" && words.length <= 5) {
    strategies.push("question_form");
  } else if (type === "comparison") {
    strategies.push("structured_comparison");
  } else {
    if (words.length <= 3) strategies.push("question_form");
  }

  return strategies;
}

/** Generate expanded query variants based on analysis. */
export function expandQueries(
  query: string,
  analysis: QueryAnalysis,
  siteFilters?: string[],
): string[] {
  const queries: string[] = [];
  const baseQuery = siteFilters?.length
    ? `${query} ${siteFilters.map((s) => `site:${s}`).join(" OR ")}`
    : query;

  for (const strategy of analysis.expansion_strategies) {
    switch (strategy) {
      case "original":
        queries.push(baseQuery);
        break;
      case "question_form":
        if (!/^(what|how|why|when|where|who|which|is|are|can|does|do)\b/i.test(query)) {
          queries.push(`what is ${query}`);
        }
        break;
      case "add_context":
        if (!/\b(documentation|docs|guide|tutorial|reference)\b/i.test(query)) {
          queries.push(`${query} documentation`);
        }
        break;
      case "add_documentation_suffix":
        queries.push(`${query} official documentation`);
        break;
      case "scholarly_rephrase":
        queries.push(`${query} research paper`);
        break;
      case "structured_comparison":
        queries.push(`${query} comparison benchmark`);
        break;
    }
  }

  return [...new Set(queries)].slice(0, 4);
}
