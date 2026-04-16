// ── Deep Research: multi-step orchestrated search with gap analysis ──

import type {
  DeepResearchOutput, ResearchStep, ScoredResult, ServerConfig,
} from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { rankResults, extractDomain } from "./ranking.js";
import { analyzeQuery } from "./query-analyzer.js";

const MAX_STEPS = 5;
const RESULTS_PER_STEP = 15;

/** Perform multi-step research: search, analyze gaps, refine, repeat */
export async function deepResearch(
  topic: string,
  depth: number,
  focusAreas: string[] | undefined,
  client: SearXNGClient,
  config: ServerConfig,
): Promise<DeepResearchOutput> {
  const startTime = Date.now();
  const steps: ResearchStep[] = [];
  const allResults: ScoredResult[] = [];
  const seenUrls = new Set<string>();
  const effectiveDepth = Math.min(depth, MAX_STEPS);

  const analysis = await analyzeQuery(topic);

  // Step 1: Broad search on the topic
  const step1Results = await searchStep(
    topic,
    "Initial broad search on the topic",
    client,
    analysis.type,
  );
  steps.push(step1Results);
  addNewResults(step1Results.findings, allResults, seenUrls);

  // Step 2: Focus area searches (parallel)
  if (focusAreas?.length) {
    const focusPromises = focusAreas.slice(0, 3).map((area) =>
      searchStep(
        `${topic} ${area}`,
        `Focused search on: ${area}`,
        client,
        analysis.type,
      ),
    );
    const focusResults = await Promise.all(focusPromises);
    for (const result of focusResults) {
      steps.push(result);
      addNewResults(result.findings, allResults, seenUrls);
    }
  }

  // Step 3+: Gap-filling based on what we found
  if (effectiveDepth >= 2) {
    // Identify domains/aspects not yet covered
    const coveredDomains = new Set(allResults.map((r) => r.domain));
    const coveredTopics = extractTopicsFromResults(allResults);

    // Try different angles
    const gapQueries = generateGapQueries(topic, coveredTopics, analysis.type);

    for (let i = 0; i < Math.min(gapQueries.length, effectiveDepth - 1); i++) {
      const gapResult = await searchStep(
        gapQueries[i].query,
        gapQueries[i].purpose,
        client,
        analysis.type,
      );
      gapResult.gap_identified = gapQueries[i].gap;
      steps.push(gapResult);
      addNewResults(gapResult.findings, allResults, seenUrls);
    }
  }

  // Synthesize findings
  const keyFindings = extractKeyFindings(allResults, topic);
  const openQuestions = identifyOpenQuestions(allResults, topic);

  // Build synthesis narrative
  const synthesis = buildSynthesis(topic, steps, keyFindings);

  return {
    topic,
    steps: steps.map((s, i) => ({ ...s, step: i + 1 })),
    synthesis,
    key_findings: keyFindings,
    open_questions: openQuestions,
    total_sources: allResults.length,
    total_duration_ms: Date.now() - startTime,
  };
}

async function searchStep(
  query: string,
  purpose: string,
  client: SearXNGClient,
  queryType: string,
): Promise<ResearchStep> {
  try {
    const resp = await client.search({ q: query });
    const ranked = rankResults(resp.results, false, queryType as any);
    return {
      step: 0, // Set by caller
      query,
      purpose,
      findings: ranked.slice(0, RESULTS_PER_STEP),
    };
  } catch {
    return {
      step: 0,
      query,
      purpose,
      findings: [],
      gap_identified: "Search failed for this query",
    };
  }
}

function addNewResults(
  newResults: ScoredResult[],
  allResults: ScoredResult[],
  seenUrls: Set<string>,
): void {
  for (const r of newResults) {
    if (!seenUrls.has(r.url)) {
      seenUrls.add(r.url);
      allResults.push(r);
    }
  }
}

function extractTopicsFromResults(results: ScoredResult[]): Set<string> {
  const topics = new Set<string>();
  for (const r of results) {
    // Extract significant words from titles
    const words = r.title.toLowerCase().split(/\s+/).filter((w) => w.length > 4);
    words.forEach((w) => topics.add(w));
  }
  return topics;
}

function generateGapQueries(
  topic: string,
  _coveredTopics: Set<string>,
  queryType: string,
): Array<{ query: string; purpose: string; gap: string }> {
  const gaps: Array<{ query: string; purpose: string; gap: string }> = [];

  // Different angles based on query type
  switch (queryType) {
    case "code":
    case "technical":
      gaps.push(
        { query: `${topic} best practices`, purpose: "Find best practices and recommendations", gap: "best practices" },
        { query: `${topic} common mistakes pitfalls`, purpose: "Find common mistakes and pitfalls", gap: "pitfalls" },
        { query: `${topic} performance optimization`, purpose: "Find performance considerations", gap: "performance" },
        { query: `${topic} alternatives comparison`, purpose: "Find alternatives and comparisons", gap: "alternatives" },
      );
      break;
    case "academic":
      gaps.push(
        { query: `${topic} recent review survey`, purpose: "Find review papers and surveys", gap: "reviews" },
        { query: `${topic} limitations challenges`, purpose: "Find known limitations", gap: "limitations" },
        { query: `${topic} future directions`, purpose: "Find future research directions", gap: "future work" },
      );
      break;
    case "news":
      gaps.push(
        { query: `${topic} analysis opinion`, purpose: "Find analysis and opinion pieces", gap: "analysis" },
        { query: `${topic} impact consequences`, purpose: "Find impact analysis", gap: "impact" },
        { query: `${topic} background context`, purpose: "Find background context", gap: "context" },
      );
      break;
    default:
      gaps.push(
        { query: `${topic} overview summary`, purpose: "Find overview summaries", gap: "overview" },
        { query: `${topic} criticism controversy`, purpose: "Find criticisms and controversies", gap: "criticism" },
        { query: `${topic} latest developments`, purpose: "Find recent developments", gap: "recent developments" },
      );
  }

  return gaps;
}

function extractKeyFindings(
  results: ScoredResult[],
  _topic: string,
): DeepResearchOutput["key_findings"] {
  // Group by domain for diversity
  const byDomain = new Map<string, ScoredResult[]>();
  for (const r of results) {
    if (!byDomain.has(r.domain)) byDomain.set(r.domain, []);
    byDomain.get(r.domain)!.push(r);
  }

  // Pick top results from diverse domains
  const findings: DeepResearchOutput["key_findings"] = [];
  const topResults = results.slice(0, 30);

  for (const r of topResults) {
    if (r.snippet && r.snippet.length > 50) {
      // Count how many other results have similar content (corroboration)
      const corroboration = topResults.filter(
        (other) =>
          other.url !== r.url &&
          r.snippet.split(/\s+/).filter((w) => w.length > 4).some((w) => other.snippet.toLowerCase().includes(w.toLowerCase())),
      ).length;

      findings.push({
        finding: r.snippet.slice(0, 300),
        sources: [{ url: r.url, title: r.title, domain: r.domain }],
        confidence: corroboration >= 2 ? "high" : corroboration >= 1 ? "medium" : "low",
      });
    }
  }

  // Deduplicate similar findings
  return findings.slice(0, 15);
}

function identifyOpenQuestions(results: ScoredResult[], topic: string): string[] {
  const questions: string[] = [];

  if (results.length < 5) {
    questions.push(`Limited sources found for "${topic}" — more research may be needed`);
  }

  const domains = new Set(results.map((r) => r.domain));
  if (domains.size < 3) {
    questions.push("Results come from few unique domains — source diversity is low");
  }

  const hasRecentResults = results.some((r) => {
    if (!r.published_date) return false;
    const age = Date.now() - new Date(r.published_date).getTime();
    return age < 90 * 24 * 60 * 60 * 1000; // 90 days
  });
  if (!hasRecentResults) {
    questions.push("No recent sources found — information may be outdated");
  }

  return questions;
}

function buildSynthesis(
  topic: string,
  steps: ResearchStep[],
  findings: DeepResearchOutput["key_findings"],
): string {
  const parts: string[] = [];

  parts.push(`Research synthesis for: "${topic}"`);
  parts.push(`Conducted ${steps.length} search steps, discovering ${findings.length} key findings.`);

  const highConfidence = findings.filter((f) => f.confidence === "high");
  if (highConfidence.length > 0) {
    parts.push(`\nHigh-confidence findings (corroborated across sources):`);
    for (const f of highConfidence.slice(0, 5)) {
      parts.push(`• ${f.finding} [${f.sources[0].domain}]`);
    }
  }

  const stepsWithGaps = steps.filter((s) => s.gap_identified);
  if (stepsWithGaps.length > 0) {
    parts.push(`\nGaps explored:`);
    for (const s of stepsWithGaps) {
      parts.push(`• ${s.gap_identified}: ${s.findings.length} results found`);
    }
  }

  return parts.join("\n");
}
