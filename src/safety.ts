// ── URL Safety: SSRF Protection ──

const PRIVATE_RANGES = [
  /^127\./,
  /^10\./,
  /^172\.(1[6-9]|2\d|3[01])\./,
  /^192\.168\./,
  /^0\./,
  /^169\.254\./,
  /^fc00:/i,
  /^fd/i,
  /^fe80:/i,
  /^::1$/,
  /^localhost$/i,
  /^0\.0\.0\.0$/,
];

const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

export interface UrlValidation {
  safe: boolean;
  reason?: string;
  url?: URL;
}

/** Validate a URL for safe fetching (SSRF protection) */
export function validateUrl(raw: string): UrlValidation {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    return { safe: false, reason: "Invalid URL format" };
  }

  // Protocol check
  if (!ALLOWED_PROTOCOLS.has(parsed.protocol)) {
    return {
      safe: false,
      reason: `Protocol '${parsed.protocol}' not allowed. Only http: and https: are permitted.`,
    };
  }

  // Check for IP-based hostnames or private ranges
  const hostname = parsed.hostname.toLowerCase();

  for (const pattern of PRIVATE_RANGES) {
    if (pattern.test(hostname)) {
      return {
        safe: false,
        reason: `Blocked: hostname '${hostname}' resolves to a private/reserved address range.`,
      };
    }
  }

  // Block common internal hostnames
  const internalNames = [
    "localhost",
    "metadata",
    "metadata.google.internal",
    "instance-data",
    "kubernetes.default",
  ];
  if (internalNames.some((n) => hostname === n || hostname.endsWith(`.${n}`))) {
    return {
      safe: false,
      reason: `Blocked: hostname '${hostname}' is an internal/metadata endpoint.`,
    };
  }

  return { safe: true, url: parsed };
}
