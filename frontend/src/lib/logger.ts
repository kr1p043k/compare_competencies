const LOG_ENDPOINT = "/api/log";

function sendLog(level: string, message: string, data?: Record<string, unknown>) {
  try {
    fetch(LOG_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level, message, data, timestamp: new Date().toISOString() }),
      keepalive: true,
    });
  } catch {}
}

let fetchOriginal: typeof globalThis.fetch | null = null;

export function initApiLogger() {
  if (fetchOriginal) return;
  fetchOriginal = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const t0 = performance.now();
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    try {
      const res = await fetchOriginal!(input, init);
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
