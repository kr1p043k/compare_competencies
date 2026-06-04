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
    const decoded = JSON.parse(atob(token.split(".")[0] + "===".slice(0, (4 - token.split(".")[0].length % 4) % 4)));
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

function authHeaders(): Record<string, string> {
  try {
    const stored = localStorage.getItem("auth");
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed.token) {
        return { Authorization: `Bearer ${parsed.token}` };
      }
    }
  } catch {}
  return {};
}

export async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers = { ...authHeaders(), ...(init?.headers as Record<string, string> || {}) };
  return nativeFetch(url, { ...init, headers });
}

/** Log frontend action to request_logs (fire-and-forget). */
export function logAction(action: string, detail?: string) {
  nativeFetch("/api/log", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ action, detail: detail || "" }),
  }).catch(() => {});
}
