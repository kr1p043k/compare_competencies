import { createContext, useContext, useState, type ReactNode } from "react";

const nativeFetch = globalThis.fetch.bind(globalThis);

interface AuthState {
  token: string | null;
  role: string | null;
  name: string | null;
  username: string | null;
}

interface AuthContextType extends AuthState {
  login: (token: string, role: string, name: string) => void;
  logout: () => void;
  isAuth: boolean;
}

const AuthContext = createContext<AuthContextType>({
  token: null, role: null, name: null, username: null,
  login: () => {}, logout: () => {}, isAuth: false,
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(() => {
    const stored = localStorage.getItem("auth");
    if (stored) {
      try { return JSON.parse(stored); } catch { /* ignore */ }
    }
    return { token: null, role: null, name: null, username: null };
  });

  const persistState = (next: AuthState) => {
    if (next.token) {
      localStorage.setItem("auth", JSON.stringify(next));
    } else {
      localStorage.removeItem("auth");
    }
  };

  const login = (token: string, role: string, name: string) => {
    const payload = token.split(".")[0];
    const decoded = JSON.parse(decodeBase64Url(payload));
    const next = { token, role, name, username: decoded.u };
    persistState(next);
    setState(next);
  };

  const logout = () => {
    const next = { token: null, role: null, name: null, username: null };
    persistState(next);
    setState(next);
  };

  return (
    <AuthContext.Provider value={{ ...state, login, logout, isAuth: !!state.token }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);

export function authHeaders(): Record<string, string> {
  const token = getToken();
  if (token) {
    document.cookie = `token=${token}; path=/; max-age=86400; SameSite=Lax`;
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

export async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers = { ...authHeaders(), ...(init?.headers as Record<string, string> || {}) };
  const token = getToken();
  let finalUrl = url;
  if (token && !url.includes("/api/auth/")) {
    finalUrl = url + (url.includes("?") ? "&" : "?") + `token=${encodeURIComponent(token)}`;
  }
  const res = await nativeFetch(finalUrl, { ...init, headers });
  if (res.status === 401 && token && !url.includes("/api/auth/login")) {
    handleUnauthorized();
  }
  return res;
}

/** Clear stale session (invalid/expired token) and bounce back to the login screen. */
export function handleUnauthorized() {
  try {
    localStorage.removeItem("auth");
  } catch {}
  document.cookie = "token=; path=/; max-age=0; SameSite=Lax";
  if (window.location.pathname !== "/") {
    window.location.assign("/");
  } else {
    window.location.reload();
  }
}

function getToken(): string | null {
  try {
    const stored = localStorage.getItem("auth");
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.token || null;
    }
  } catch {}
  return null;
}

/** Decode a base64url string (RFC 4648 §5) using browser atob(). */
function decodeBase64Url(input: string): string {
  let b64 = input.replace(/-/g, "+").replace(/_/g, "/");
  while (b64.length % 4 !== 0) b64 += "=";
  return atob(b64);
}

/** Log frontend action to request_logs (fire-and-forget). */
export function logAction(action: string, detail?: string) {
  nativeFetch("/api/log", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ action, detail: detail || "" }),
  }).catch(() => {});
}
