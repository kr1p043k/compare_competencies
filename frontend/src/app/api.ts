import { authHeaders } from "../lib/auth";

const API_BASE = "/api";

export async function api(endpoint: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers as Record<string, string> | undefined),
      ...authHeaders(),
    },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}
