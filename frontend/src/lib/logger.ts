const LOG_ENDPOINT = "/api/log";

function sendLog(level: string, message: string, data?: Record<string, unknown>) {
  const f = fetchOriginal ?? globalThis.fetch;
  try {
    f(LOG_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level, message, data, timestamp: new Date().toISOString() }),
      keepalive: true,
    });
  } catch {}
}

function getAuthToken(): string | null {
  try {
    const stored = localStorage.getItem("auth");
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.token || null;
    }
  } catch {}
  return null;
}

let fetchOriginal: typeof globalThis.fetch | null = null;

export function initApiLogger() {
  if (fetchOriginal) return;
  const orig = globalThis.fetch.bind(globalThis);
  fetchOriginal = orig;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const t0 = performance.now();
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const token = getAuthToken();
    const origHeaders: Record<string, string> = {};
    if (init?.headers) {
      const h = init.headers;
      if (h instanceof Headers) {
        h.forEach((v, k) => { origHeaders[k] = v; });
      } else if (Array.isArray(h)) {
        h.forEach(([k, v]) => { origHeaders[k] = v; });
      } else {
        Object.assign(origHeaders, h);
      }
    }
    if (token && !url.includes("/api/auth/")) {
      origHeaders["Authorization"] = `Bearer ${token}`;
    }
    const newInit: RequestInit = { ...init, headers: origHeaders };
    try {
      const res = await orig(input, newInit);
      const elapsed = performance.now() - t0;
      if (elapsed > 1000 || !res.ok) {
        sendLog(elapsed > 1000 ? "warn" : "error", `API ${elapsed > 1000 ? "slow" : "error"}`, {
          url, method: init?.method || "GET", elapsed_ms: Math.round(elapsed), status: res.status });
      }
      return res;
    } catch (err) {
      const elapsed = performance.now() - t0;
      sendLog("error", "API failed", { url, method: init?.method || "GET", elapsed_ms: Math.round(elapsed), error: String(err) });
      throw err;
    }
  };
}
