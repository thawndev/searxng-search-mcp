// ── UI Inspiration Engine: design search → curated images + metadata for LLM analysis ──

import type { SearXNGRawResult, ServerConfig } from "./types.js";
import { SearXNGClient } from "./searxng-client.js";
import { validateUrl } from "./safety.js";
import { embedTexts, isModelReady } from "./embeddings.js";
import { LRUCache } from "./cache.js";

// ── Types ──

export interface UIInspireParams {
  query: string;
  style?: string;
  platform?: "mobile" | "web" | "tablet" | "desktop";
  components?: string[];
  max_images?: number;
  mode?: "thumbnails" | "links_only" | "inspect";
  framework?: string;
  safesearch?: 0 | 1 | 2;
}

export interface UIImageResult {
  title: string;
  source_url: string;
  image_url: string;
  thumbnail_url?: string;
  source_domain: string;
  resolution?: string;
  format?: string;
  design_relevance: number; // 0-1
}

export interface UIInspireOutput {
  query: string;
  expanded_queries: string[];
  images: UIImageResult[];
  fetched_images: Array<{
    image: UIImageResult;
    data: string;       // base64
    mimeType: string;
    sizeBytes: number;
  }>;
  code_references: Array<{
    title: string;
    url: string;
    snippet: string;
    domain: string;
  }>;
  metadata: {
    total_candidates: number;
    engines_used: string[];
    mode: string;
    duration_ms: number;
  };
}

// ── Domain Reputation for UI/Design Sources ──

const DESIGN_DOMAIN_SCORES: Record<string, number> = {
  "dribbble.com": 1.0,
  "behance.net": 1.0,
  "figma.com": 0.95,
  "awwwards.com": 0.95,
  "mobbin.com": 0.95,
  "screenlane.com": 0.9,
  "uigarage.net": 0.9,
  "collectui.com": 0.9,
  "pttrns.com": 0.85,
  "pinterest.com": 0.75,
  "deviantart.com": 0.6,
  "unsplash.com": 0.5,
  "pexels.com": 0.4,
  "flickr.com": 0.3,
};

// Penalize non-UI content
const PENALTY_PATTERNS = [
  /mockup\s*bundle/i, /template\s*pack/i, /wallpaper/i,
  /stock\s*photo/i, /poster/i, /flyer/i, /brochure/i,
  /logo\s*design/i, /icon\s*pack/i, /font/i, /banner\s*ad/i,
];

// Boost UI-specific signals
const BOOST_PATTERNS = [
  /\b(ui|ux)\b/i, /\bscreen\b/i, /\bapp\b/i, /\bdashboard\b/i,
  /\binterface\b/i, /\bdesign\s*system\b/i, /\bcomponent\b/i,
  /\blayout\b/i, /\bprototype\b/i, /\bwireframe\b/i,
  /\bonboarding\b/i, /\blogin\b/i, /\bsettings\b/i,
  /\bcase\s*study\b/i, /\bconcept\b/i,
];

// ── Image Cache ──

const imageCache = new LRUCache<{ data: string; mimeType: string; sizeBytes: number }>(50, 600_000);
export { imageCache };

// ── Max byte size for fetched images ──

const MAX_IMAGE_BYTES = 512 * 1024; // 512KB per image
const MAX_TOTAL_BYTES = 1.5 * 1024 * 1024; // 1.5MB total payload
const ALLOWED_IMAGE_TYPES = new Set([
  "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
]);

// ── Query Expansion ──

function expandDesignQuery(params: UIInspireParams): string[] {
  const { query, style, platform, components } = params;
  const queries: string[] = [];

  // Primary: direct query + UI context
  const platformTag = platform ? `${platform} ` : "";
  const styleTag = style ? ` ${style}` : "";
  queries.push(`${platformTag}${query}${styleTag} UI design`);

  // Design platform targeted
  queries.push(`${query} UI UX design dribbble behance`);

  // Component-specific queries
  if (components?.length) {
    const componentStr = components.slice(0, 3).join(" ");
    queries.push(`${query} ${componentStr} UI component design`);
  }

  // Style-specific if provided
  if (style) {
    queries.push(`${query} ${style} style interface design`);
  }

  return queries;
}

// ── Image Scoring ──

function scoreImageResult(
  result: SearXNGRawResult,
  query: string,
  semanticScores?: Map<string, number>,
): number {
  let score = 0;
  const title = (result.title || "").toLowerCase();
  const domain = extractDomainFromUrl(result.url || result.img_src || "");

  // Domain reputation (0-1, weight: 40%)
  const domainScore = Object.entries(DESIGN_DOMAIN_SCORES).reduce((best, [d, s]) => {
    return domain.includes(d) ? Math.max(best, s) : best;
  }, 0.2); // default 0.2 for unknown domains
  score += domainScore * 0.4;

  // Penalty patterns (subtract up to 0.3)
  const penalties = PENALTY_PATTERNS.filter(p => p.test(title)).length;
  score -= Math.min(penalties * 0.15, 0.3);

  // Boost patterns (add up to 0.2)
  const boosts = BOOST_PATTERNS.filter(p => p.test(title)).length;
  score += Math.min(boosts * 0.05, 0.2);

  // Resolution scoring (weight: 15%)
  if (result.resolution) {
    const match = result.resolution.match(/(\d+)\s*[×x]\s*(\d+)/);
    if (match) {
      const [w, h] = [parseInt(match[1]), parseInt(match[2])];
      const pixels = w * h;
      // Prefer screen-like resolutions (not tiny icons, not massive posters)
      if (pixels >= 200000 && pixels <= 8000000) {
        score += 0.15;
      } else if (pixels >= 50000) {
        score += 0.08;
      }
    }
  }

  // Semantic similarity (weight: 25%)
  if (semanticScores) {
    const semScore = semanticScores.get(result.title || "") || 0;
    score += semScore * 0.25;
  }

  // Multi-engine consensus bonus
  if (result.engines && result.engines.length > 1) {
    score += Math.min(result.engines.length * 0.03, 0.1);
  }

  return Math.max(0, Math.min(1, score));
}

function extractDomainFromUrl(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return "";
  }
}

// ── Safe Image Fetch ──

async function fetchImageSafe(
  url: string,
  config: ServerConfig,
): Promise<{ data: string; mimeType: string; sizeBytes: number } | null> {
  // Check cache first
  const cached = imageCache.get(url);
  if (cached) return cached;

  // SSRF validation
  const validation = validateUrl(url);
  if (!validation.safe) return null;

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(config.requestTimeout, 10000));

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; searxng-search-mcp)",
        Accept: "image/png,image/jpeg,image/webp,image/gif,*/*;q=0.1",
      },
      redirect: "follow",
    });
    clearTimeout(timer);

    if (!response.ok) return null;

    // Validate content type
    const contentType = (response.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
    if (!ALLOWED_IMAGE_TYPES.has(contentType)) return null;

    // Check content length before downloading
    const declaredSize = parseInt(response.headers.get("content-length") || "0");
    if (declaredSize > MAX_IMAGE_BYTES) return null;

    // Stream with byte cap
    const reader = response.body?.getReader();
    if (!reader) return null;

    const chunks: Uint8Array[] = [];
    let totalBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > MAX_IMAGE_BYTES) {
        reader.cancel();
        return null;
      }
      chunks.push(value);
    }

    // Combine chunks and convert to base64
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }

    const base64 = Buffer.from(combined).toString("base64");
    const result = { data: base64, mimeType: contentType, sizeBytes: totalBytes };

    // Cache it
    imageCache.set(url, result);
    return result;
  } catch {
    return null;
  }
}

// ── Code Reference Search ──

async function searchCodeReferences(
  query: string,
  framework: string,
  client: SearXNGClient,
): Promise<UIInspireOutput["code_references"]> {
  try {
    const codeQuery = `${query} ${framework} UI component code example site:github.com OR site:dev.to OR site:medium.com OR site:stackoverflow.com`;
    const response = await client.search({
      q: codeQuery,
      categories: "general",
      safesearch: 1,
    });

    return response.results
      .filter(r => {
        const domain = extractDomainFromUrl(r.url);
        return domain.includes("github.com") || domain.includes("dev.to") ||
               domain.includes("medium.com") || domain.includes("stackoverflow.com") ||
               domain.includes("npmjs.com") || domain.includes("expo.dev");
      })
      .slice(0, 5)
      .map(r => ({
        title: r.title || "",
        url: r.url,
        snippet: (r.content || "").substring(0, 300),
        domain: extractDomainFromUrl(r.url),
      }));
  } catch {
    return [];
  }
}

// ── Main Pipeline ──

export async function uiInspire(
  params: UIInspireParams,
  client: SearXNGClient,
  config: ServerConfig,
): Promise<UIInspireOutput> {
  const startTime = Date.now();
  const mode = params.mode || "thumbnails";
  const maxImages = Math.min(params.max_images || 3, mode === "inspect" ? 1 : 6);
  const safesearch = params.safesearch ?? 1;

  // 1. Expand queries
  const expandedQueries = expandDesignQuery(params);

  // 2. Search for images across all expanded queries in parallel
  const searchPromises = expandedQueries.map(q =>
    client.search({ q, categories: "images", safesearch }).catch(() => null)
  );

  // Optionally search for code references
  const codePromise = params.framework
    ? searchCodeReferences(params.query, params.framework, client)
    : Promise.resolve([]);

  const [searchResults, codeReferences] = await Promise.all([
    Promise.all(searchPromises),
    codePromise,
  ]);

  // 3. Collect and deduplicate all image results
  const seenUrls = new Set<string>();
  const allResults: SearXNGRawResult[] = [];

  for (const response of searchResults) {
    if (!response) continue;
    for (const result of response.results) {
      const imgUrl = result.img_src || result.thumbnail_src;
      if (!imgUrl || seenUrls.has(imgUrl)) continue;
      seenUrls.add(imgUrl);
      allResults.push(result);
    }
  }

  // 4. Semantic scoring (if model ready)
  let semanticScores: Map<string, number> | undefined;
  if (isModelReady() && allResults.length > 0) {
    try {
      const titles = allResults.slice(0, 50).map(r => r.title || "");
      const [queryEmbed, ...titleEmbeds] = await embedTexts([params.query, ...titles]);
      if (queryEmbed && titleEmbeds.length > 0) {
        semanticScores = new Map();
        for (let i = 0; i < titleEmbeds.length; i++) {
          const sim = cosineSimilarity(queryEmbed, titleEmbeds[i]);
          semanticScores.set(titles[i], sim);
        }
      }
    } catch {
      // Non-critical — fall back to heuristic scoring only
    }
  }

  // 5. Score and rank
  const scored = allResults.map(r => ({
    result: r,
    score: scoreImageResult(r, params.query, semanticScores),
  }));
  scored.sort((a, b) => b.score - a.score);

  // 6. Build ranked image list
  const topImages: UIImageResult[] = scored
    .slice(0, Math.max(maxImages * 3, 20)) // keep more candidates for fetch fallback
    .map(({ result, score }) => ({
      title: result.title || "Untitled",
      source_url: result.url || "",
      image_url: result.img_src || "",
      thumbnail_url: result.thumbnail_src,
      source_domain: extractDomainFromUrl(result.url || result.img_src || ""),
      resolution: result.resolution,
      format: result.img_format,
      design_relevance: Math.round(score * 100) / 100,
    }));

  // 7. Fetch images based on mode
  const fetchedImages: UIInspireOutput["fetched_images"] = [];

  if (mode !== "links_only") {
    let totalFetchedBytes = 0;
    const fetchTargetCount = mode === "inspect" ? 1 : maxImages;

    for (const img of topImages) {
      if (fetchedImages.length >= fetchTargetCount) break;
      if (totalFetchedBytes >= MAX_TOTAL_BYTES) break;

      // For thumbnails mode, prefer thumbnail_url; for inspect, prefer full image
      const fetchUrl = mode === "inspect"
        ? (img.image_url || img.thumbnail_url)
        : (img.thumbnail_url || img.image_url);

      if (!fetchUrl) continue;

      const fetched = await fetchImageSafe(fetchUrl, config);
      if (fetched) {
        totalFetchedBytes += fetched.sizeBytes;
        fetchedImages.push({
          image: img,
          data: fetched.data,
          mimeType: fetched.mimeType,
          sizeBytes: fetched.sizeBytes,
        });
      }
    }
  }

  // Collect engines used
  const enginesUsed = [...new Set(
    searchResults
      .filter(Boolean)
      .flatMap(r => r!.results.map(res => res.engine))
      .filter(Boolean)
  )];

  return {
    query: params.query,
    expanded_queries: expandedQueries,
    images: topImages.slice(0, Math.max(maxImages, 10)), // return more metadata even if fewer fetched
    fetched_images: fetchedImages,
    code_references: codeReferences,
    metadata: {
      total_candidates: allResults.length,
      engines_used: enginesUsed,
      mode,
      duration_ms: Date.now() - startTime,
    },
  };
}

// ── Output Formatter ──

export function formatUIInspireOutput(output: UIInspireOutput): Array<{ type: "text" | "image"; text?: string; data?: string; mimeType?: string }> {
  const content: Array<{ type: "text" | "image"; text?: string; data?: string; mimeType?: string }> = [];

  // Header
  const lines: string[] = [];
  lines.push(`**UI Inspiration** | ${output.metadata.mode} mode | ${output.images.length} designs found | ${output.metadata.engines_used.join(", ")} | ${output.metadata.duration_ms}ms`);
  lines.push(`*Queries: ${output.expanded_queries.join(" | ")}*`);
  lines.push("");

  // Image results with metadata
  lines.push("### Design References");
  for (let i = 0; i < output.images.length; i++) {
    const img = output.images[i];
    const res = img.resolution ? ` [${img.resolution}]` : "";
    const fmt = img.format ? ` (${img.format})` : "";
    const rel = ` — relevance: ${img.design_relevance}`;
    lines.push(`[${i + 1}] **${img.title}** — ${img.source_domain}${res}${fmt}${rel}`);
    lines.push(`    Source: ${img.source_url}`);
    lines.push(`    Image: ${img.image_url}`);
  }

  // Push text header first
  content.push({ type: "text", text: lines.join("\n") });

  // Push fetched images as MCP image content blocks
  for (const fetched of output.fetched_images) {
    // Add a text label before each image
    content.push({
      type: "text",
      text: `\n**${fetched.image.title}** (${fetched.image.source_domain}, ${Math.round(fetched.sizeBytes / 1024)}KB${fetched.image.resolution ? `, ${fetched.image.resolution}` : ""})`,
    });
    content.push({
      type: "image",
      data: fetched.data,
      mimeType: fetched.mimeType,
    });
  }

  // Code references
  if (output.code_references.length > 0) {
    const codeLines: string[] = ["\n### Code References"];
    for (const ref of output.code_references) {
      codeLines.push(`- **${ref.title}** — ${ref.domain}`);
      codeLines.push(`  ${ref.url}`);
      if (ref.snippet) codeLines.push(`  ${ref.snippet.substring(0, 200)}`);
    }
    content.push({ type: "text", text: codeLines.join("\n") });
  }

  // Fallback text for clients that can't render images
  if (output.fetched_images.length > 0) {
    content.push({
      type: "text",
      text: `\n*${output.fetched_images.length} image(s) attached above. If images don't render, use the source URLs to view them.*`,
    });
  }

  return content;
}

// ── Cosine similarity helper ──

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
