// ── Claim Verification: multi-query evidence gathering and classification ──

import type {
  VerifyClaimOutput, EvidenceItem, ServerConfig,
} from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { rankResults, extractDomain } from "./ranking.js";
import { verifyCache, LRUCache } from "./cache.js";

const STOP_WORDS = new Set([
  "is", "are", "was", "were", "the", "a", "an", "of", "in", "to", "for",
  "and", "or", "that", "this", "it", "with", "from", "by", "on", "at",
  "as", "but", "not", "has", "have", "had", "be", "been", "will", "would",
  "could", "should", "can", "may", "might", "do", "does", "did",
]);

function claimToQueries(claim: string, context?: string): string[] {
  const queries: string[] = [];

  queries.push(claim);

  // Exact phrase search for short claims
  if (claim.length < 120) {
    queries.push(`"${claim}"`);
  }

  // Extract significant terms
  const significant = claim
    .split(/\s+/)
    .filter((w) => w.length > 3 && !STOP_WORDS.has(w.toLowerCase()))
    .join(" ");
  if (significant && significant !== claim && significant.split(/\s+/).length >= 2) {
    queries.push(significant);
  }

  if (context) {
    queries.push(`${context} ${claim}`);
  }

  // Try negation to find contradicting evidence
  queries.push(`${claim} false OR incorrect OR myth OR debunked`);

  return [...new Set(queries)].slice(0, 5);
}

const SUPPORT_PHRASES = [
  "confirmed", "verified", "true", "correct", "accurate", "indeed",
  "according to", "studies show", "research confirms", "evidence suggests",
  "data shows", "proven", "established",
];

const CONTRADICT_PHRASES = [
  "not true", "false", "incorrect", "myth", "debunked", "misleading",
  "inaccurate", "wrong", "no evidence", "contrary to", "disproven",
  "unfounded", "baseless", "misrepresentation", "misinformation",
  "claim is false", "actually", "however", "in reality",
];

function classifyEvidence(
  snippet: string,
  claim: string,
): { stance: "supports" | "contradicts" | "neutral"; score: number } {
  const lowerSnippet = snippet.toLowerCase();
  const lowerClaim = claim.toLowerCase();

  const claimTerms = lowerClaim
    .split(/\s+/)
    .filter((w) => w.length > 3 && !STOP_WORDS.has(w))
    .slice(0, 10);

  const matchCount = claimTerms.filter((t) => lowerSnippet.includes(t)).length;
  const matchRatio = claimTerms.length > 0 ? matchCount / claimTerms.length : 0;

  // Score contradiction and support signals
  let supportScore = 0;
  let contradictScore = 0;

  for (const phrase of SUPPORT_PHRASES) {
    if (lowerSnippet.includes(phrase)) supportScore += 0.3;
  }
  for (const phrase of CONTRADICT_PHRASES) {
    if (lowerSnippet.includes(phrase)) contradictScore += 0.3;
  }

  // Relevance matters: if snippet doesn't relate to claim, it's neutral
  if (matchRatio < 0.2) return { stance: "neutral", score: matchRatio };

  if (contradictScore > supportScore && contradictScore > 0.2) {
    return { stance: "contradicts", score: contradictScore };
  }
  if (matchRatio > 0.4 && supportScore >= contradictScore) {
    return { stance: "supports", score: matchRatio + supportScore };
  }

  return { stance: "neutral", score: matchRatio };
}

export async function verifyClaim(
  claim: string,
  context: string | undefined,
  client: SearXNGClient,
  config: ServerConfig,
): Promise<VerifyClaimOutput> {
  // Check cache
  const cacheKey = LRUCache.makeKey({ claim, context });
  const cached = verifyCache.get(cacheKey) as VerifyClaimOutput | undefined;
  if (cached) return cached;

  const queries = claimToQueries(claim, context);

  const allResults = [];
  const searchPromises = queries.map((q) =>
    client.search({ q }).catch(() => null),
  );

  const responses = await Promise.all(searchPromises);
  for (const resp of responses) {
    if (resp) allResults.push(...resp.results);
  }

  const ranked = rankResults(allResults, false, "factual");

  // Classify evidence
  const evidence: EvidenceItem[] = ranked.slice(0, 20).map((r) => {
    const classification = classifyEvidence(r.snippet, claim);
    return {
      url: r.url,
      title: r.title,
      snippet: r.snippet,
      domain: r.domain,
      supports_claim: classification.stance,
      engines: r.engines,
      relevance_score: r.relevance_score,
    };
  });

  const supporting = evidence.filter((e) => e.supports_claim === "supports");
  const contradicting = evidence.filter((e) => e.supports_claim === "contradicts");
  const neutral = evidence.filter((e) => e.supports_claim === "neutral");
  const uniqueDomains = new Set(evidence.map((e) => e.domain)).size;

  let verdict: VerifyClaimOutput["verdict"];
  let confidence: VerifyClaimOutput["confidence"];

  if (evidence.length < 3) {
    verdict = "insufficient_evidence";
    confidence = "low";
  } else if (supporting.length >= 3 && contradicting.length === 0) {
    verdict = "supported";
    confidence = uniqueDomains >= 3 ? "high" : "medium";
  } else if (contradicting.length >= 2 && supporting.length === 0) {
    verdict = "contradicted";
    confidence = uniqueDomains >= 3 ? "high" : "medium";
  } else if (supporting.length > 0 && contradicting.length > 0) {
    verdict = "mixed";
    confidence = "medium";
  } else if (supporting.length > 0) {
    verdict = "supported";
    confidence = supporting.length >= 2 ? "medium" : "low";
  } else if (contradicting.length > 0) {
    verdict = "contradicted";
    confidence = "low";
  } else {
    verdict = "insufficient_evidence";
    confidence = "low";
  }

  const summaryParts: string[] = [
    `Claim: "${claim}"`,
    `Verdict: ${verdict} (${confidence} confidence).`,
    `Analyzed ${evidence.length} sources from ${uniqueDomains} unique domains using ${queries.length} search queries.`,
  ];
  if (supporting.length > 0) summaryParts.push(`${supporting.length} source(s) support the claim.`);
  if (contradicting.length > 0) summaryParts.push(`${contradicting.length} source(s) contradict the claim.`);

  const output: VerifyClaimOutput = {
    claim,
    verdict,
    confidence,
    evidence,
    queries_used: queries,
    summary: summaryParts.join(" "),
    metadata: {
      supporting_count: supporting.length,
      contradicting_count: contradicting.length,
      neutral_count: neutral.length,
      sources_checked: evidence.length,
      unique_domains: uniqueDomains,
    },
  };

  verifyCache.set(cacheKey, output);
  return output;
}
