// ── Content Fetcher: URL fetch + HTML → clean text + multi-format support ──

import * as cheerio from "cheerio";
import type { FetchPageOutput, ServerConfig } from "./types.js";
import { validateUrl } from "./safety.js";
import { extractDomain } from "./ranking.js";
import { fetchCache, LRUCache } from "./cache.js";

const SELECTORS_TO_REMOVE = [
  "script", "style", "noscript", "iframe", "svg", "canvas",
  "nav", "header:not(article header)", "footer:not(article footer)",
  ".ad", ".ads", ".advertisement", ".sidebar", ".cookie-banner",
  ".cookie-consent", ".popup", ".modal", ".overlay",
  "[role='navigation']", "[role='banner']", "[role='contentinfo']",
  ".social-share", ".comments", ".comment-section", ".related-posts",
  ".newsletter", ".signup-form", ".promo", ".sponsored",
  "#cookie-notice", "#gdpr", ".gdpr",
];

/** Fetch a URL and extract clean text content */
export async function fetchPage(
  url: string,
  config: ServerConfig,
  maxLength?: number,
): Promise<FetchPageOutput> {
  const validation = validateUrl(url);
  if (!validation.safe) {
    throw new Error(`URL blocked: ${validation.reason}`);
  }

  // Check cache
  const cacheKey = LRUCache.makeKey({ url, maxLength });
  const cached = fetchCache.get(cacheKey) as FetchPageOutput | undefined;
  if (cached) return { ...cached, warning: "(cached)" };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.requestTimeout);

  let response: Response;
  try {
    response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; searxng-search-mcp)",
        Accept: "text/html,application/xhtml+xml,application/json,text/plain,text/markdown,application/xml,*/*;q=0.8",
      },
      redirect: "follow",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Fetch timed out after ${config.requestTimeout}ms for ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText} fetching ${url}`);
  }

  const contentType = response.headers.get("content-type") || "";
  const rawBody = await response.text();
  const effectiveMax = Math.min(maxLength ?? config.maxFetchSize, config.maxFetchSize);

  let title = "";
  let content = "";
  let description: string | undefined;
  let author: string | undefined;
  let publishedDate: string | undefined;
  let headings: string[] = [];
  let linksCount = 0;
  let language: string | undefined;

  if (contentType.includes("text/html") || contentType.includes("xhtml")) {
    const result = extractFromHtml(rawBody);
    title = result.title;
    content = result.content;
    description = result.description;
    author = result.author;
    publishedDate = result.publishedDate;
    headings = result.headings;
    linksCount = result.linksCount;
    language = result.language;
  } else if (contentType.includes("application/json")) {
    try {
      const parsed = JSON.parse(rawBody);
      title = "JSON Document";
      content = JSON.stringify(parsed, null, 2);
    } catch {
      content = rawBody;
    }
  } else if (contentType.includes("text/markdown") || url.endsWith(".md")) {
    title = rawBody.match(/^#\s+(.+)/m)?.[1] || "Markdown Document";
    content = rawBody;
  } else if (contentType.includes("application/xml") || contentType.includes("text/xml")) {
    title = "XML Document";
    content = rawBody;
  } else {
    title = "Text Document";
    content = rawBody;
  }

  // Smart truncation: try to break at paragraph boundaries
  let warning: string | undefined;
  if (content.length > effectiveMax) {
    const truncateAt = content.lastIndexOf("\n\n", effectiveMax);
    const breakPoint = truncateAt > effectiveMax * 0.7 ? truncateAt : effectiveMax;
    content = content.slice(0, breakPoint) + "\n\n[... content truncated ...]";
    warning = `Content truncated from ${rawBody.length} to ~${breakPoint} characters`;
  }

  const wordCount = content.split(/\s+/).filter(Boolean).length;

  const output: FetchPageOutput = {
    url,
    title,
    content,
    content_length: content.length,
    content_type: contentType,
    fetched_at: new Date().toISOString(),
    metadata: {
      description,
      author,
      published_date: publishedDate,
      domain: extractDomain(url),
      headings: headings.slice(0, 20),
      links_count: linksCount,
      word_count: wordCount,
      language,
    },
    warning,
  };

  fetchCache.set(cacheKey, output);
  return output;
}

function extractFromHtml(rawBody: string) {
  const $ = cheerio.load(rawBody);

  const title = $("title").first().text().trim() ||
    $('meta[property="og:title"]').attr("content") ||
    $("h1").first().text().trim() || "";

  const description = $('meta[name="description"]').attr("content") ||
    $('meta[property="og:description"]').attr("content");

  const author = $('meta[name="author"]').attr("content") ||
    $('meta[property="article:author"]').attr("content") ||
    $('[rel="author"]').first().text().trim() || undefined;

  const publishedDate = $('meta[property="article:published_time"]').attr("content") ||
    $('meta[name="date"]').attr("content") ||
    $("time[datetime]").first().attr("datetime") || undefined;

  const language = $("html").attr("lang") || $('meta[http-equiv="content-language"]').attr("content") || undefined;

  // Collect headings before removal
  const headings: string[] = [];
  $("h1, h2, h3, h4").each((_, el) => {
    const text = $(el).text().trim();
    if (text) headings.push(text);
  });

  const linksCount = $("a[href]").length;

  // Remove non-content elements
  SELECTORS_TO_REMOVE.forEach((sel) => {
    try { $(sel).remove(); } catch { /* ignore invalid selectors */ }
  });

  // Extract main content
  let mainEl = $("article").first();
  if (!mainEl.length) mainEl = $("main").first();
  if (!mainEl.length) mainEl = $('[role="main"]').first();
  if (!mainEl.length) mainEl = $(".post-content, .article-content, .entry-content, .content").first();
  if (!mainEl.length) mainEl = $("body");

  // Build structured content
  const parts: string[] = [];
  mainEl.find("p, h1, h2, h3, h4, h5, h6, li, pre, code, blockquote, table, dl, dt, dd").each((_, el) => {
    const tag = (el as unknown as { tagName: string }).tagName;
    const text = $(el).text().trim();
    if (!text) return;

    if (tag.startsWith("h")) {
      const level = parseInt(tag[1]);
      parts.push(`\n${"#".repeat(level)} ${text}\n`);
    } else if (tag === "li") {
      parts.push(`• ${text}`);
    } else if (tag === "pre") {
      // Try to detect language from class
      const langClass = $(el).attr("class") || $(el).find("code").attr("class") || "";
      const langMatch = langClass.match(/language-(\w+)|lang-(\w+)|(\w+)-code/);
      const lang = langMatch ? (langMatch[1] || langMatch[2] || langMatch[3]) : "";
      parts.push(`\`\`\`${lang}\n${text}\n\`\`\``);
    } else if (tag === "code" && !$(el).parent("pre").length) {
      parts.push(`\`${text}\``);
    } else if (tag === "blockquote") {
      parts.push(`> ${text}`);
    } else if (tag === "table") {
      // Simple table extraction
      const rows: string[] = [];
      $(el).find("tr").each((_, tr) => {
        const cells = $(tr).find("td, th").map((__, cell) => $(cell).text().trim()).get();
        rows.push(`| ${cells.join(" | ")} |`);
      });
      if (rows.length > 0) parts.push(rows.join("\n"));
    } else if (tag === "dt") {
      parts.push(`**${text}**`);
    } else if (tag === "dd") {
      parts.push(`  ${text}`);
    } else {
      parts.push(text);
    }
  });

  let content = parts.filter(Boolean).join("\n\n");

  // Fallback
  if (content.length < 100) {
    content = mainEl.text().replace(/\s+/g, " ").trim();
  }

  return { title, content, description, author, publishedDate, headings, linksCount, language };
}

/** Batch fetch multiple URLs concurrently */
export async function fetchMultiplePages(
  urls: string[],
  config: ServerConfig,
  maxLength?: number,
): Promise<Array<{ url: string; result?: FetchPageOutput; error?: string }>> {
  const results = await Promise.all(
    urls.map(async (url) => {
      try {
        const result = await fetchPage(url, config, maxLength);
        return { url, result };
      } catch (err) {
        return { url, error: err instanceof Error ? err.message : String(err) };
      }
    }),
  );
  return results;
}
