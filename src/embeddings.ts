// ── Semantic Intelligence Layer ──
// bge-small-en-v1.5 embeddings for: (1) zero-shot query classification, (2) RRF result reranking.
// No hardcoded regex patterns — query understanding is purely neural.

import type { ScoredResult, QueryType } from "./types.js";

// ─── Types ───

interface EmbeddingModel {
  (texts: string[], options: { pooling: string; normalize: boolean }): Promise<{ tolist: () => number[][] }>;
}

interface EmbeddingCacheEntry {
  vector: number[];
  accessedAt: number;
}

export interface NeuralClassification {
  type: QueryType;
  confidence: number;       // top-2 gap
  secondaryType: QueryType;
  timeSensitive: boolean;
  suggestedDepth: "quick" | "normal" | "deep";
  suggestedTopic: "code" | "news" | "academic" | "general";
}

// ─── Configuration ───

const MODEL_ID = "Xenova/bge-small-en-v1.5";
const MAX_CACHE_ENTRIES = 1000;
const MODEL_LOAD_TIMEOUT_MS = 30_000;
const EMBED_TIMEOUT_MS = 5_000;
const RERANK_TOP_N = 30; // Only rerank top N results for performance
const RRF_K = 60; // RRF constant — standard value from literature
const TITLE_WEIGHT = 0.7;
const SNIPPET_WEIGHT = 0.3;
const MAX_SNIPPET_CHARS = 300; // Truncate snippets before embedding

// ─── Singleton Model Loading ───

let modelPromise: Promise<EmbeddingModel> | null = null;
let modelReady = false;
let modelFailed = false;
let lastFailTime = 0;
const RETRY_AFTER_FAIL_MS = 60_000; // Retry model load after 1 minute

function getModel(): Promise<EmbeddingModel> {
  // If model failed recently, don't retry yet
  if (modelFailed && Date.now() - lastFailTime < RETRY_AFTER_FAIL_MS) {
    return Promise.reject(new Error("Embedding model recently failed to load"));
  }

  if (!modelPromise) {
    modelPromise = loadModel();
  }
  return modelPromise;
}

async function loadModel(): Promise<EmbeddingModel> {
  try {
    // Dynamic import to avoid blocking startup if not available
    const { pipeline, env } = await import("@huggingface/transformers");

    // Configure cache directory
    env.cacheDir = "./.cache/models";

    const result = await Promise.race([
      pipeline("feature-extraction", MODEL_ID, {
        dtype: "q8" as any,
      } as any),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Model load timeout")), MODEL_LOAD_TIMEOUT_MS),
      ),
    ]);

    modelReady = true;
    modelFailed = false;
    console.error(`[searxng-mcp] Embedding model loaded: ${MODEL_ID}`);
    return result as unknown as EmbeddingModel;
  } catch (err) {
    modelFailed = true;
    lastFailTime = Date.now();
    modelPromise = null; // Allow retry
    console.error(`[searxng-mcp] Embedding model failed to load: ${err}`);
    throw err;
  }
}

// ─── Embedding Cache ───

const embeddingCache = new Map<string, EmbeddingCacheEntry>();

function getCachedEmbedding(text: string): number[] | undefined {
  const entry = embeddingCache.get(text);
  if (entry) {
    entry.accessedAt = Date.now();
    return entry.vector;
  }
  return undefined;
}

function setCachedEmbedding(text: string, vector: number[]): void {
  // Evict oldest entries if cache is full
  if (embeddingCache.size >= MAX_CACHE_ENTRIES) {
    let oldestKey = "";
    let oldestTime = Infinity;
    for (const [key, entry] of embeddingCache) {
      if (entry.accessedAt < oldestTime) {
        oldestTime = entry.accessedAt;
        oldestKey = key;
      }
    }
    if (oldestKey) embeddingCache.delete(oldestKey);
  }
  embeddingCache.set(text, { vector, accessedAt: Date.now() });
}

// ─── Core Embedding Functions ───

/** Embed multiple texts in one batch. Returns one vector per text. */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const model = await getModel();

  // Check cache — find which texts need embedding
  const results: (number[] | null)[] = texts.map((t) => getCachedEmbedding(t) ?? null);
  const uncachedIndices: number[] = [];
  const uncachedTexts: string[] = [];

  for (let i = 0; i < texts.length; i++) {
    if (!results[i]) {
      uncachedIndices.push(i);
      uncachedTexts.push(texts[i]);
    }
  }

  // Embed uncached texts
  if (uncachedTexts.length > 0) {
    const output = await Promise.race([
      model(uncachedTexts, { pooling: "cls", normalize: true }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Embedding timeout")), EMBED_TIMEOUT_MS),
      ),
    ]);

    const vectors = output.tolist();
    for (let i = 0; i < uncachedIndices.length; i++) {
      results[uncachedIndices[i]] = vectors[i];
      setCachedEmbedding(uncachedTexts[i], vectors[i]);
    }
  }

  return results as number[][];
}

/** Cosine similarity between two normalized vectors. */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  // Vectors are already normalized by the model, so dot product = cosine similarity
  return dot;
}

// ─── Neural Query Classification ───
// Zero-shot classification using archetype embeddings.
// Each query type is defined by natural language descriptions (prototypes).
// Classification = max cosine similarity over prototypes per type.

const TYPE_PROTOTYPES: Record<string, string[]> = {
  code: [
    "programming code function API library package module npm pip cargo",
    "error debugging exception stack trace bug crash TypeError fix",
    "configuration setup install config yaml json settings file import",
    "react vue angular nextjs tailwind nativewind expo typescript python rust",
    "docker kubernetes deployment CI/CD pipeline terraform devops AWS",
    "database SQL MongoDB PostgreSQL Redis query migration schema ORM",
    "CSS styling global.css tailwind.config webpack vite postcss sass",
    "git commit branch merge pull request repository version control",
    "testing jest vitest playwright cypress unit integration test mock",
  ],
  news: [
    "breaking news announcement today politics economy business report",
    "sports entertainment celebrity trending viral social media",
    "government policy election regulation law court ruling verdict",
  ],
  academic: [
    "research paper scientific study journal peer reviewed citation DOI",
    "thesis hypothesis methodology experiment results conclusion abstract",
    "machine learning deep learning neural network algorithm model theory",
  ],
  howto: [
    "how to tutorial guide step by step instructions walkthrough beginner",
    "best practice recommended approach proper way example getting started",
  ],
  troubleshooting: [
    "fix solve resolve debug not working broken problem issue crash",
    "error help stuck cannot unable failing exception runtime compile",
    "why does why doesn't why isn't something wrong unexpected behavior",
  ],
  comparison: [
    "versus vs comparison difference between pros cons alternative which",
    "benchmark performance review tradeoffs advantages disadvantages",
  ],
  definition: [
    "what is definition meaning explanation concept overview introduction",
    "who is biography background history about describe",
  ],
  factual: [
    "when where how many how much statistics data population number",
    "capital founded invented discovered percentage rate count year",
  ],
  general: [
    "search find information learn understand question answer",
  ],
};

const FRESHNESS_PROTOTYPES = {
  timeSensitive: [
    "latest breaking news today current events now trending update",
    "price stock market weather forecast score live real-time election",
    "what happened this week this month announcement released launch",
  ],
  timeless: [
    "documentation guide tutorial reference specification standard RFC",
    "concept theory definition how to configure install setup basics",
    "history overview introduction fundamentals architecture design",
  ],
};

const DEPTH_PROTOTYPES: Record<string, string[]> = {
  quick: [
    "simple quick answer definition lookup basic fact what is meaning",
    "short question one sentence answer yes or no",
  ],
  normal: [
    "moderate information overview comparison general question explain",
    "facts about list of summary brief description",
  ],
  deep: [
    "complex technical detailed code implementation debugging configuration",
    "comprehensive guide tutorial full documentation example walkthrough",
    "in-depth analysis investigation multi-step solution architecture",
  ],
};

const TOPIC_PROTOTYPES: Record<string, string[]> = {
  code: [
    "programming software development code framework library API SDK",
    "error debugging configuration install deploy build compile",
    "React Vue Angular Docker Kubernetes TypeScript Python Rust database",
    "npm pip package module import require setup config schema ORM",
    "CSS styling HTML component function class method fix resolve",
  ],
  news: [
    "news events politics business economy sports entertainment trending",
    "breaking announcement released launched today this week current",
  ],
  academic: [
    "research paper scientific study mathematics physics biology chemistry",
    "peer reviewed journal citation DOI thesis methodology experiment",
  ],
  general: [
    "general information question answer learn find out about",
    "population capital founded when where who what meaning definition",
  ],
};

// Pre-computed archetype embeddings (initialized at warmup)
let typeEmbeddings: Map<string, number[][]> | null = null;
let freshnessEmbeddings: { timeSensitive: number[][]; timeless: number[][] } | null = null;
let depthEmbeddings: Map<string, number[][]> | null = null;
let topicEmbeddings: Map<string, number[][]> | null = null;
let archetypesReady = false;

/** Pre-compute all archetype embeddings. Called during warmup. */
async function initArchetypes(): Promise<void> {
  if (archetypesReady) return;

  // Collect all texts for one big batch embed
  const allTexts: string[] = [];
  const segments: { name: string; start: number; texts: string[] }[] = [];

  for (const [type, texts] of Object.entries(TYPE_PROTOTYPES)) {
    segments.push({ name: `type:${type}`, start: allTexts.length, texts });
    allTexts.push(...texts);
  }
  for (const [key, texts] of Object.entries(FRESHNESS_PROTOTYPES)) {
    segments.push({ name: `fresh:${key}`, start: allTexts.length, texts });
    allTexts.push(...texts);
  }
  for (const [depth, texts] of Object.entries(DEPTH_PROTOTYPES)) {
    segments.push({ name: `depth:${depth}`, start: allTexts.length, texts });
    allTexts.push(...texts);
  }
  for (const [topic, texts] of Object.entries(TOPIC_PROTOTYPES)) {
    segments.push({ name: `topic:${topic}`, start: allTexts.length, texts });
    allTexts.push(...texts);
  }

  const embeddings = await embedTexts(allTexts);

  // Distribute into typed maps
  typeEmbeddings = new Map();
  depthEmbeddings = new Map();
  topicEmbeddings = new Map();

  for (const seg of segments) {
    const vecs = embeddings.slice(seg.start, seg.start + seg.texts.length);
    if (seg.name.startsWith("type:")) {
      typeEmbeddings.set(seg.name.slice(5), vecs);
    } else if (seg.name === "fresh:timeSensitive") {
      freshnessEmbeddings = freshnessEmbeddings || { timeSensitive: [], timeless: [] };
      freshnessEmbeddings.timeSensitive = vecs;
    } else if (seg.name === "fresh:timeless") {
      freshnessEmbeddings = freshnessEmbeddings || { timeSensitive: [], timeless: [] };
      freshnessEmbeddings.timeless = vecs;
    } else if (seg.name.startsWith("depth:")) {
      depthEmbeddings.set(seg.name.slice(6), vecs);
    } else if (seg.name.startsWith("topic:")) {
      topicEmbeddings.set(seg.name.slice(6), vecs);
    }
  }

  archetypesReady = true;
  console.error(`[searxng-mcp] Archetype embeddings initialized (${allTexts.length} prototypes)`);
}

/** Max similarity of query embedding against a set of prototype embeddings. */
function maxSimilarity(queryEmb: number[], prototypes: number[][]): number {
  let max = -1;
  for (const proto of prototypes) {
    const sim = cosineSimilarity(queryEmb, proto);
    if (sim > max) max = sim;
  }
  return max;
}

/** Classify a map of labels → prototype embeddings, returning sorted scores. */
function classifyAgainst(
  queryEmb: number[],
  labelMap: Map<string, number[][]>,
): { label: string; score: number }[] {
  const scores: { label: string; score: number }[] = [];
  for (const [label, protos] of labelMap) {
    scores.push({ label, score: maxSimilarity(queryEmb, protos) });
  }
  scores.sort((a, b) => b.score - a.score);
  return scores;
}

/**
 * Neural zero-shot query classification.
 * Returns null if model/archetypes not ready (caller should use fallback).
 */
export async function classifyQuery(query: string): Promise<NeuralClassification | null> {
  if (!modelReady || !archetypesReady || !typeEmbeddings || !freshnessEmbeddings || !depthEmbeddings || !topicEmbeddings) {
    return null;
  }

  try {
    const [queryEmb] = await embedTexts([query]);

    // 1. Query type classification
    const typeScores = classifyAgainst(queryEmb, typeEmbeddings);
    const bestType = typeScores[0];
    const secondType = typeScores[1];
    const confidence = bestType.score - secondType.score;

    // 2. Freshness classification (with margin)
    const tsSim = maxSimilarity(queryEmb, freshnessEmbeddings.timeSensitive);
    const tlSim = maxSimilarity(queryEmb, freshnessEmbeddings.timeless);
    // Require clear margin — code/config queries with "latest" should NOT be time-sensitive
    const timeSensitive = tsSim > tlSim + 0.08;

    // 3. Depth classification
    const depthScores = classifyAgainst(queryEmb, depthEmbeddings);
    const suggestedDepth = depthScores[0].label as "quick" | "normal" | "deep";

    // 4. Topic classification
    const topicScores = classifyAgainst(queryEmb, topicEmbeddings);
    const suggestedTopic = topicScores[0].label as "code" | "news" | "academic" | "general";

    return {
      type: bestType.label as QueryType,
      confidence,
      secondaryType: secondType.label as QueryType,
      timeSensitive,
      suggestedDepth,
      suggestedTopic,
    };
  } catch (err) {
    console.error(`[searxng-mcp] Neural classification failed: ${err}`);
    return null;
  }
}

// ─── Reciprocal Rank Fusion ───

interface RankedItem {
  index: number;
  keywordRank: number;
  semanticRank: number;
  consensusRank: number;
  rrfScore: number;
}

/**
 * Reciprocal Rank Fusion — combines multiple rank lists into one.
 * RRF score = Σ 1/(k + rank_i) for each ranking signal.
 * k=60 is the standard constant from the RRF paper.
 */
function computeRRF(
  keywordRank: number,
  semanticRank: number,
  consensusRank: number,
): number {
  return (
    1 / (RRF_K + keywordRank) +
    1 / (RRF_K + semanticRank) +
    1 / (RRF_K + consensusRank)
  );
}

// ─── Semantic Reranking ───

/**
 * Semantically rerank search results using embedding similarity + RRF.
 *
 * Process:
 * 1. Take top N results (by keyword score)
 * 2. Embed query, all titles, and all snippets
 * 3. Compute semantic similarity (0.7 title + 0.3 snippet)
 * 4. Rank by semantic similarity
 * 5. Rank by engine count (consensus)
 * 6. Fuse keyword rank + semantic rank + consensus rank via RRF
 * 7. Return reranked results with updated relevance_score
 *
 * Falls back to keyword-only ranking on any embedding failure.
 */
export async function semanticRerank(
  query: string,
  results: ScoredResult[],
  topN: number = RERANK_TOP_N,
): Promise<ScoredResult[]> {
  if (results.length === 0) return results;

  try {
    // Only rerank top N for performance
    const toRerank = results.slice(0, Math.min(topN, results.length));
    const rest = results.slice(toRerank.length);

    // Prepare texts — separate title and snippet
    const titles = toRerank.map((r) => r.title || "");
    const snippets = toRerank.map((r) =>
      (r.snippet || "").slice(0, MAX_SNIPPET_CHARS),
    );

    // Batch embed: [query, ...titles, ...snippets]
    const allTexts = [query, ...titles, ...snippets];
    const embeddings = await embedTexts(allTexts);

    const queryEmb = embeddings[0];
    const titleEmbs = embeddings.slice(1, 1 + toRerank.length);
    const snippetEmbs = embeddings.slice(1 + toRerank.length);

    // Compute semantic scores
    const semanticScores = toRerank.map((_, i) => {
      const titleSim = cosineSimilarity(queryEmb, titleEmbs[i]);
      const snippetSim = snippetEmbs[i]
        ? cosineSimilarity(queryEmb, snippetEmbs[i])
        : 0;
      return titleSim * TITLE_WEIGHT + snippetSim * SNIPPET_WEIGHT;
    });

    // Build rank lists
    // Keyword rank: already sorted by keyword score (index = rank)
    // Semantic rank: sort by semantic score
    const semanticOrder = toRerank
      .map((_, i) => ({ index: i, score: semanticScores[i] }))
      .sort((a, b) => b.score - a.score);
    const semanticRankMap = new Map<number, number>();
    semanticOrder.forEach((item, rank) =>
      semanticRankMap.set(item.index, rank),
    );

    // Consensus rank: sort by engine count descending
    const consensusOrder = toRerank
      .map((_, i) => ({ index: i, count: toRerank[i].engine_count }))
      .sort((a, b) => b.count - a.count);
    const consensusRankMap = new Map<number, number>();
    consensusOrder.forEach((item, rank) =>
      consensusRankMap.set(item.index, rank),
    );

    // Compute RRF scores
    const rrfItems: RankedItem[] = toRerank.map((_, i) => ({
      index: i,
      keywordRank: i, // Already sorted by keyword score
      semanticRank: semanticRankMap.get(i)!,
      consensusRank: consensusRankMap.get(i)!,
      rrfScore: computeRRF(
        i,
        semanticRankMap.get(i)!,
        consensusRankMap.get(i)!,
      ),
    }));

    // Sort by RRF score descending
    rrfItems.sort((a, b) => b.rrfScore - a.rrfScore);

    // Build reranked results with updated scores
    const reranked: ScoredResult[] = rrfItems.map((item) => ({
      ...toRerank[item.index],
      relevance_score: item.rrfScore,
      semantic_score: semanticScores[item.index],
    }));

    return [...reranked, ...rest];
  } catch (err) {
    console.error(`[searxng-mcp] Semantic reranking failed, using keyword ranking: ${err}`);
    return results; // Graceful degradation
  }
}

// ─── Utilities ───

/** Warmup: pre-load the embedding model and initialize archetype embeddings. */
export async function warmupEmbeddings(): Promise<boolean> {
  try {
    await getModel();
    await initArchetypes();
    return true;
  } catch {
    return false;
  }
}

/** Check if the embedding model is ready. */
export function isModelReady(): boolean {
  return modelReady;
}

/** Get embedding cache stats. */
export function getEmbeddingStats(): {
  modelReady: boolean;
  modelFailed: boolean;
  cacheSize: number;
  maxCacheSize: number;
} {
  return {
    modelReady,
    modelFailed,
    cacheSize: embeddingCache.size,
    maxCacheSize: MAX_CACHE_ENTRIES,
  };
}
