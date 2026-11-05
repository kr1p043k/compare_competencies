import { handleUnauthorized } from "./auth";

const LOG_ENDPOINT = "/api/log";

// Сервер лимитирует /api/log (20/мин) и обрезает message до 2000 символов.
// Без дедупликации пачка медленных запросов (прогрев, тяжёлый эндпоинт) даёт
// поток одинаковых логов и 429. Одинаковые сообщения шлём не чаще раза в 5 c.
const lastSent: Record<string, number> = {};

function sendLog(level: string, message: string, data?: Record<string, unknown>) {
  const key = `${level}:${message}:${String(data?.url ?? "")}`;
  const now = Date.now();
  if (lastSent[key] && now - lastSent[key] < 5000) return;
  lastSent[key] = now;
  const f = fetchOriginal ?? globalThis.fetch;
  try {
    f(LOG_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level, message: message.slice(0, 2000), data, timestamp: new Date(now).toISOString() }),
      keepalive: true,
    });
  } catch {}
}

let fetchOriginal: typeof globalThis.fetch | null = null;

export function initApiLogger() {
  if (fetchOriginal) return;
  const orig = globalThis.fetch.bind(globalThis);
  fetchOriginal = orig;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const t0 = performance.now();
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
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
    // Токена в JS нет: сессию прикладывает браузер из HttpOnly-cookie. Явно
    // просим её отправить, чтобы перехватчик не менял поведение fetch.
    const newInit: RequestInit = { ...init, headers: origHeaders, credentials: init?.credentials ?? "same-origin" };
    try {
      const res = await orig(input, newInit);
      const elapsed = performance.now() - t0;
      if (res.status === 401 && !url.includes("/api/auth/login")) {
        handleUnauthorized();
      }
      if (elapsed > 1000 || !res.ok) {
        // "warning", не "warn": сервер валидирует level строгим Literal
        // (debug/info/warning/error) и отвечал 422 на "warn".
        sendLog(elapsed > 1000 ? "warning" : "error", `API ${elapsed > 1000 ? "slow" : "error"}`, {
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
