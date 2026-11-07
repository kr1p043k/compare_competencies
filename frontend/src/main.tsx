import React from 'react';
import { createRoot } from "react-dom/client";
import App from "./app/App.tsx";
import { AuthProvider, setSession } from "./lib/auth.tsx";
import { extractHubToken } from "./lib/hubToken";
import "./styles/index.css";
import { reportError } from "./lib/errors";
import ErrorBoundary from "./design/ErrorBoundary";

// React загружен (версию смотреть в package.json)

/**
 * Вход с хаба ЮФУ: hub.sfedu.ru открывает наш сайт с JWT - в query
 * (`?token=...`) или во фрагменте (`#token=...`). Забираем токен из адреса,
 * сразу чистим его (чтобы не светился в истории/логах), обмениваем на наш
 * токен через /api/auth/sso и сохраняем сессию.
 */
async function performHubSso(): Promise<void> {
  const { token: hubToken, search, hash } = extractHubToken(
    window.location.search,
    window.location.hash,
  );
  if (!hubToken) return;

  window.history.replaceState({}, document.title, window.location.pathname + search + hash);

  try {
    const res = await fetch("/api/auth/sso", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: hubToken }),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) {
      let detail = `SSO не прошёл (HTTP ${res.status})`;
      try {
        const j = await res.json();
        if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch (err) { reportError("main.tsx:36", err); }
      localStorage.removeItem("auth");
      throw new Error(detail);
    }
    const data = await res.json();
    if (!data.token) {
      // Сервер ответил статусом 2FA вместо токена: cookie не выставлена,
      // значит вход не состоялся. Просим войти обычным способом.
      throw new Error(
        data.status === "totp_setup_required" || data.status === "totp_required"
          ? "Вход через хаб требует двухфакторной аутентификации. Войдите по логину и паролю."
          : "Сервер не выдал сессию",
      );
    }
    setSession({ role: data.role, name: data.name, username: data.username });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Не удалось подтвердить токен хаба";
    console.error("[SSO] Ошибка авторизации:", msg);
    // Показываем причину на экране входа: раньше ошибка уходила только в
    // консоль, и неудачный вход через хаб выглядел как «ничего не произошло».
    try { sessionStorage.setItem("ssoError", msg); } catch (err) { reportError("main.tsx:ssoError", err); }
    window.dispatchEvent(new CustomEvent("sso-error", { detail: msg }));
  }
}

async function bootstrap() {
  await performHubSso();

  const root = document.getElementById("root");
  if (!root) {
    throw new Error("Root element not found");
  }

  createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundary scope="root">
        <AuthProvider>
          <App />
        </AuthProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
}

bootstrap();
