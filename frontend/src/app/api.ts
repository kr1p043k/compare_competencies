/**
 * Тонкая обёртка над `lib/api.ts` для вызовов вида `api("/path")`.
 *
 * Раньше здесь был отдельный fetch с ручной проверкой `res.ok`, который
 * бросал сырой текст ответа. Теперь вся обработка (разбор detail, ApiError,
 * 401 -> logout) живёт в одном месте.
 */
import { request } from "../lib/api";

const API_BASE = "/api";

export function api<T = unknown>(endpoint: string, options: RequestInit = {}): Promise<T> {
  return request<T>(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...((options.headers as Record<string, string>) ?? {}),
    },
    body: options.body as BodyInit | undefined,
  });
}