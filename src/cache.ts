// ── LRU Cache with TTL, stats, and namespaced keys ──

import type { CacheEntry } from "./types.js";

export class LRUCache<T> {
  private cache = new Map<string, CacheEntry<T>>();
  private maxSize: number;
  private defaultTtl: number;
  private hits = 0;
  private misses = 0;

  constructor(maxSize: number = 500, defaultTtlMs: number = 300_000) {
    this.maxSize = maxSize;
    this.defaultTtl = defaultTtlMs;
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) {
      this.misses++;
      return undefined;
    }

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.misses++;
      return undefined;
    }

    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, entry);
    this.hits++;
    return entry.data;
  }

  set(key: string, data: T, ttlMs?: number): void {
    // Evict oldest entries if at capacity
    while (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
      else break;
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMs ?? this.defaultTtl,
    });
  }

  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  invalidate(pattern: string): number {
    let count = 0;
    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  get size(): number {
    return this.cache.size;
  }

  get stats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: this.hits + this.misses > 0
        ? Math.round((this.hits / (this.hits + this.misses)) * 100)
        : 0,
    };
  }

  static makeKey(parts: Record<string, unknown>): string {
    const sorted = Object.keys(parts)
      .sort()
      .reduce(
        (acc, k) => {
          if (parts[k] !== undefined && parts[k] !== null) {
            acc[k] = parts[k];
          }
          return acc;
        },
        {} as Record<string, unknown>,
      );
    return JSON.stringify(sorted);
  }
}

// Shared cache instances
export const searchCache = new LRUCache<unknown>(500, 300_000);
export const fetchCache = new LRUCache<unknown>(200, 600_000);
export const verifyCache = new LRUCache<unknown>(100, 600_000);
