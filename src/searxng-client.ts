// ── SearXNG API Client with retry, concurrency, multi-instance failover ──

import type { SearXNGResponse, ServerConfig, RetryConfig } from "./types.js";

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig,
  label: string,
): Promise<T> {
  let lastError: Error | undefined;
  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < config.maxRetries) {
        const delay = Math.min(
          config.baseDelayMs * Math.pow(2, attempt) + Math.random() * 200,
          config.maxDelayMs,
        );
        console.error(
          `[searxng-mcp] ${label} attempt ${attempt + 1} failed: ${lastError.message}. Retrying in ${Math.round(delay)}ms...`,
        );
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError!;
}

// ── Instance Health Tracking ──

interface InstanceHealth {
  url: string;
  lastSuccess: number;
  lastError: number;
  consecutiveErrors: number;
  avgLatencyMs: number;
  totalQueries: number;
  totalErrors: number;
  suspendedUntil: number;  // timestamp — if > now, skip this instance
}

function createHealth(url: string): InstanceHealth {
  return {
    url,
    lastSuccess: 0,
    lastError: 0,
    consecutiveErrors: 0,
    avgLatencyMs: 0,
    totalQueries: 0,
    totalErrors: 0,
    suspendedUntil: 0,
  };
}

// ── Multi-Instance Client ──

export class SearXNGClient {
  private instances: InstanceHealth[];
  private apiKey?: string;
  private timeout: number;
  private retry: RetryConfig;
  private activeRequests = 0;
  private maxConcurrent: number;
  private requestQueue: Array<() => void> = [];
  private hedgeDelayMs: number;
  private minResultThreshold: number;

  constructor(config: ServerConfig) {
    // Parse comma-separated URLs for multi-instance support
    const urls = config.searxngUrl
      .split(",")
      .map((u) => u.trim().replace(/\/+$/, ""))
      .filter(Boolean);

    this.instances = urls.map(createHealth);
    this.apiKey = config.apiKey;
    this.timeout = config.requestTimeout;
    this.retry = config.retry;
    this.maxConcurrent = config.maxConcurrentRequests;
    this.hedgeDelayMs = 2000; // start hedge after 2s
    this.minResultThreshold = 3; // if < 3 results, try another instance
  }

  private async acquireSlot(): Promise<void> {
    if (this.activeRequests < this.maxConcurrent) {
      this.activeRequests++;
      return;
    }
    return new Promise((resolve) => {
      this.requestQueue.push(() => {
        this.activeRequests++;
        resolve();
      });
    });
  }

  private releaseSlot(): void {
    this.activeRequests--;
    const next = this.requestQueue.shift();
    if (next) next();
  }

  /** Pick the best instance based on health */
  private selectInstance(): InstanceHealth {
    const now = Date.now();

    // Filter out suspended instances
    const available = this.instances.filter((i) => i.suspendedUntil < now);
    if (available.length === 0) {
      // All suspended — pick least-suspended one and unsuspend it
      this.instances.sort((a, b) => a.suspendedUntil - b.suspendedUntil);
      this.instances[0].suspendedUntil = 0;
      return this.instances[0];
    }

    // Score: lower is better
    // Prefer: low latency, low error rate, recent success
    available.sort((a, b) => {
      const scoreA =
        a.avgLatencyMs * 0.3 +
        a.consecutiveErrors * 500 +
        (now - a.lastSuccess) * 0.001;
      const scoreB =
        b.avgLatencyMs * 0.3 +
        b.consecutiveErrors * 500 +
        (now - b.lastSuccess) * 0.001;
      return scoreA - scoreB;
    });

    return available[0];
  }

  /** Pick a different instance for hedging */
  private selectHedgeInstance(primary: InstanceHealth): InstanceHealth | null {
    const now = Date.now();
    const available = this.instances.filter(
      (i) => i.url !== primary.url && i.suspendedUntil < now,
    );
    if (available.length === 0) return null;
    // Pick the one with fewest errors
    available.sort((a, b) => a.consecutiveErrors - b.consecutiveErrors);
    return available[0];
  }

  private recordSuccess(instance: InstanceHealth, latencyMs: number): void {
    instance.lastSuccess = Date.now();
    instance.consecutiveErrors = 0;
    instance.totalQueries++;
    instance.avgLatencyMs =
      instance.totalQueries === 1
        ? latencyMs
        : instance.avgLatencyMs * 0.8 + latencyMs * 0.2;
  }

  private recordError(instance: InstanceHealth): void {
    instance.lastError = Date.now();
    instance.consecutiveErrors++;
    instance.totalErrors++;
    instance.totalQueries++;

    // Suspend instance if too many consecutive errors
    if (instance.consecutiveErrors >= 5) {
      // Exponential backoff: 30s, 60s, 120s, max 600s
      const suspendTime = Math.min(
        30000 * Math.pow(2, instance.consecutiveErrors - 5),
        600000,
      );
      instance.suspendedUntil = Date.now() + suspendTime;
      console.error(
        `[searxng-mcp] Instance ${instance.url} suspended for ${Math.round(suspendTime / 1000)}s (${instance.consecutiveErrors} consecutive errors)`,
      );
    }
  }

  /** Execute a search request against a specific instance */
  private async searchInstance(
    instance: InstanceHealth,
    params: {
      q: string;
      categories?: string;
      engines?: string;
      language?: string;
      pageno?: number;
      time_range?: string;
      safesearch?: number;
    },
  ): Promise<SearXNGResponse> {
    const startTime = Date.now();
    try {
      const result = await retryWithBackoff(
        async () => {
          const url = new URL(`${instance.url}/search`);
          url.searchParams.set("format", "json");
          url.searchParams.set("q", params.q);

          if (params.categories) url.searchParams.set("categories", params.categories);
          if (params.engines) url.searchParams.set("engines", params.engines);
          if (params.language) url.searchParams.set("language", params.language);
          if (params.pageno) url.searchParams.set("pageno", String(params.pageno));
          if (params.time_range) url.searchParams.set("time_range", params.time_range);
          if (params.safesearch !== undefined) url.searchParams.set("safesearch", String(params.safesearch));

          const headers: Record<string, string> = {
            Accept: "application/json",
            "User-Agent": "searxng-search-mcp",
          };
          if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), this.timeout);

          try {
            const response = await fetch(url.toString(), {
              method: "GET",
              headers,
              signal: controller.signal,
            });

            if (!response.ok) {
              const body = await response.text().catch(() => "");
              throw new Error(
                `SearXNG ${response.status} ${response.statusText}${body ? `: ${body.slice(0, 200)}` : ""}`,
              );
            }

            return (await response.json()) as SearXNGResponse;
          } catch (error) {
            if (error instanceof Error && error.name === "AbortError") {
              throw new Error(`SearXNG request timed out after ${this.timeout}ms`);
            }
            throw error;
          } finally {
            clearTimeout(timer);
          }
        },
        this.retry,
        `search("${params.q.slice(0, 50)}")@${new URL(instance.url).hostname}`,
      );

      this.recordSuccess(instance, Date.now() - startTime);
      return result;
    } catch (err) {
      this.recordError(instance);
      throw err;
    }
  }

  /** Main search method — hedged failover across instances */
  async search(params: {
    q: string;
    categories?: string;
    engines?: string;
    language?: string;
    pageno?: number;
    time_range?: string;
    safesearch?: number;
  }): Promise<SearXNGResponse> {
    await this.acquireSlot();
    try {
      const primary = this.selectInstance();

      // If only one instance, just use it directly
      if (this.instances.length <= 1) {
        return await this.searchInstance(primary, params);
      }

      // Hedged failover: start primary, if slow/bad start secondary
      return await new Promise<SearXNGResponse>((resolve, reject) => {
        let resolved = false;
        let primaryDone = false;
        let hedgeDone = false;
        let primaryError: Error | null = null;
        let hedgeError: Error | null = null;

        // Primary request
        this.searchInstance(primary, params)
          .then((result) => {
            primaryDone = true;
            if (!resolved) {
              // If enough results, resolve immediately
              if (result.results.length >= this.minResultThreshold) {
                resolved = true;
                resolve(result);
              } else if (hedgeDone) {
                // Primary had few results and hedge is done too
                resolved = true;
                resolve(result);
              }
              // Otherwise wait for hedge
            }
          })
          .catch((err) => {
            primaryDone = true;
            primaryError = err instanceof Error ? err : new Error(String(err));
            if (hedgeDone && !resolved) {
              resolved = true;
              if (hedgeError) {
                reject(primaryError); // both failed
              }
            }
          });

        // Hedge request after delay
        const hedge = this.selectHedgeInstance(primary);
        if (!hedge) {
          // No hedge available — just wait for primary
          const fallbackCheck = setInterval(() => {
            if (primaryDone && !resolved) {
              clearInterval(fallbackCheck);
              resolved = true;
              if (primaryError) reject(primaryError);
            }
          }, 50);
          return;
        }

        setTimeout(() => {
          if (resolved) return;

          this.searchInstance(hedge, params)
            .then((result) => {
              hedgeDone = true;
              if (!resolved && result.results.length >= this.minResultThreshold) {
                resolved = true;
                resolve(result);
              } else if (primaryDone && !resolved) {
                resolved = true;
                // Use whichever had results
                resolve(result);
              }
            })
            .catch((err) => {
              hedgeDone = true;
              hedgeError = err instanceof Error ? err : new Error(String(err));
              if (primaryDone && !resolved) {
                resolved = true;
                if (primaryError) reject(primaryError);
              }
            });
        }, this.hedgeDelayMs);

        // Overall timeout
        setTimeout(() => {
          if (!resolved) {
            resolved = true;
            reject(new Error("All instances timed out"));
          }
        }, this.timeout * 2);
      });
    } finally {
      this.releaseSlot();
    }
  }

  /** Search with automatic pagination */
  async searchPaginated(
    params: {
      q: string;
      categories?: string;
      engines?: string;
      language?: string;
      time_range?: string;
      safesearch?: number;
    },
    maxPages: number = 2,
  ): Promise<SearXNGResponse> {
    const firstPage = await this.search({ ...params, pageno: 1 });
    if (maxPages <= 1 || firstPage.results.length < 5) return firstPage;

    const additionalPages: Promise<SearXNGResponse | null>[] = [];
    for (let page = 2; page <= maxPages; page++) {
      additionalPages.push(
        this.search({ ...params, pageno: page }).catch(() => null),
      );
    }

    const pages = await Promise.all(additionalPages);
    for (const page of pages) {
      if (page) {
        firstPage.results.push(...page.results);
        firstPage.suggestions.push(...page.suggestions);
      }
    }

    return firstPage;
  }

  async autocomplete(q: string): Promise<string[]> {
    const instance = this.selectInstance();
    const url = new URL(`${instance.url}/autocompleter`);
    url.searchParams.set("q", q);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch(url.toString(), {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return [];
      const data = (await response.json()) as Array<string | [string, string]>;
      return data.map((item) => (Array.isArray(item) ? item[0] : item));
    } catch {
      return [];
    } finally {
      clearTimeout(timer);
    }
  }

  async healthCheck(): Promise<{
    healthy: boolean;
    instances: Array<{
      url: string;
      reachable: boolean;
      latencyMs: number;
      health: InstanceHealth;
    }>;
  }> {
    const checks = await Promise.all(
      this.instances.map(async (instance) => {
        const start = Date.now();
        try {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 5000);
          const response = await fetch(`${instance.url}/`, {
            signal: controller.signal,
          });
          clearTimeout(timer);
          return {
            url: instance.url,
            reachable: response.ok,
            latencyMs: Date.now() - start,
            health: { ...instance },
          };
        } catch {
          return {
            url: instance.url,
            reachable: false,
            latencyMs: Date.now() - start,
            health: { ...instance },
          };
        }
      }),
    );

    return {
      healthy: checks.some((c) => c.reachable),
      instances: checks,
    };
  }

  /** Get instance stats for diagnostics */
  get instanceStats(): Array<{
    url: string;
    queries: number;
    errors: number;
    avgLatencyMs: number;
    suspended: boolean;
  }> {
    const now = Date.now();
    return this.instances.map((i) => ({
      url: i.url,
      queries: i.totalQueries,
      errors: i.totalErrors,
      avgLatencyMs: Math.round(i.avgLatencyMs),
      suspended: i.suspendedUntil > now,
    }));
  }

  get url(): string {
    return this.instances[0]?.url ?? "";
  }
}
