import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { reportError } from "./errors";

const nativeFetch = globalThis.fetch.bind(globalThis);

/** Ключ несекретного признака «сессия была активна».
 *
 *  Токен здесь НЕ хранится: он живёт только в HttpOnly-cookie, которую ставит
 *  сервер. localStorage читает любой скрипт на странице, поэтому токен там
 *  означал бы угон сессии при одной XSS. Флаг нужен лишь чтобы первая
 *  отрисовка не мигала экраном входа; истину всё равно даёт /auth/me.
 */
const HINT_KEY = "sessionHint";

interface AuthState {
  role: string | null;
  name: string | null;
  username: string | null;
  authed: boolean;
}

interface AuthContextType extends AuthState {
  /** token принимается для совместимости с вызывающим кодом и НЕ сохраняется. */
  login: (token: string | null, role: string, name: string) => void;
  logout: () => void;
  isAuth: boolean;
}

const EMPTY: AuthState = { role: null, name: null, username: null, authed: false };

const AuthContext = createContext<AuthContextType>({
  ...EMPTY,
  login: () => {},
  logout: () => {},
  isAuth: false,
});

function setHint(on: boolean) {
  try {
    if (on) localStorage.setItem(HINT_KEY, "1");
    else localStorage.removeItem(HINT_KEY);
  } catch (err) {
    reportError("lib/auth.tsx:hint", err);
  }
}

function hadSession(): boolean {
  try {
    return localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return false;
  }
}

/** Подписчики на внешние изменения сессии (hub SSO), вне дерева React. */
const sessionListeners = new Set<(next: AuthState) => void>();

function notify(next: AuthState) {
  for (const listener of sessionListeners) listener(next);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(() => ({
    ...EMPTY,
    authed: hadSession(),
  }));

  // setSession() вызывается из main.tsx до/вне React-дерева (hub SSO):
  // подписываемся, чтобы состояние обновилось без перезагрузки.
  useEffect(() => {
    sessionListeners.add(setState);
    return () => {
      sessionListeners.delete(setState);
    };
  }, []);

  // После перезагрузки cookie остаётся, состояние React - нет. /auth/me
  // отвечает 200 только при живой сессии в БД, поэтому он и есть источник
  // истины: роль, имя и username берём оттуда, а не из токена на клиенте.
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const res = await nativeFetch("/api/auth/me", { credentials: "same-origin" });
        if (!alive) return;
        if (!res.ok) {
          setHint(false);
          setState(EMPTY);
          return;
        }
        const me = (await res.json()) as {
          role?: string;
          name?: string;
          username?: string;
        };
        setHint(true);
        setState({
          role: me.role ?? null,
          name: me.name ?? null,
          username: me.username ?? null,
          authed: true,
        });
      } catch (err) {
        if (alive) reportError("lib/auth.tsx:restore", err);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const login = (_token: string | null, role: string, name: string) => {
    setHint(true);
    setState({ role, name, username: null, authed: true });
    // username и точная роль приходят из БД: добираем, не блокируя вход.
    void (async () => {
      try {
        const res = await nativeFetch("/api/auth/me", { credentials: "same-origin" });
        if (!res.ok) return;
        const me = (await res.json()) as {
          role?: string;
          name?: string;
          username?: string;
        };
        setState({
          role: me.role ?? role,
          name: me.name ?? name,
          username: me.username ?? null,
          authed: true,
        });
      } catch (err) {
        reportError("lib/auth.tsx:login-refresh", err);
      }
    })();
  };

  const logout = () => {
    setHint(false);
    setState(EMPTY);
  };

  return (
    <AuthContext.Provider value={{ ...state, login, logout, isAuth: state.authed }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);

/** Заголовки авторизации.
 *
 *  Токена в JS больше нет - сессию прикладывает браузер из HttpOnly-cookie,
 *  поэтому достаточно пустого объекта. Функция оставлена, чтобы не менять
 *  все точки вызова и явно показывать, что запрос авторизован.
 */
export function authHeaders(): Record<string, string> {
  return {};
}

/** Сессия недействительна: сбросить локальное состояние и вернуться на вход. */
export function handleUnauthorized() {
  setHint(false);
  try {
    localStorage.removeItem("auth");
  } catch (err) {
    reportError("lib/auth.tsx:handleUnauthorized", err);
  }
  // Не перезагружаем страницу: любой повторный 401 на "/" запустил бы reload
  // снова, и так по кругу (страница «вечно обновлялась» при заходе). Вместо
  // этого сбрасываем React-состояние сессии - экран входа отрисуется сам.
  notify(EMPTY);
  if (window.location.pathname !== "/") {
    window.location.assign("/");
  }
}

/** Записать сессию после внешнего входа (hub SSO).
 *
 *  Cookie уже поставил сервер; здесь только помечаем состояние, чтобы UI не
 *  ждал следующей перезагрузки. Подписчики - активные AuthProvider-ы.
 */
export function setSession(session: { role: string; name: string; username?: string | null }) {
  setHint(true);
  notify({
    role: session.role,
    name: session.name,
    username: session.username ?? null,
    authed: true,
  });
}

/** Log frontend action to request_logs (fire-and-forget). */
export function logAction(action: string, detail?: string) {
  nativeFetch("/api/log", {
    method: "POST",
    credentials: "same-origin",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ action, detail: detail || "" }),
  }).catch((err) => reportError("lib/auth.tsx", err));
}
