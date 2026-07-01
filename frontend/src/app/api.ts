import { authHeaders } from "../lib/auth";

const API_BASE = "/api";

export async function api(endpoint: string, options?: RequestInit) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...authHeaders(),
    ...(options?.headers as Record<string, string> | undefined),
  };
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}
