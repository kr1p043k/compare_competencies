import React from 'react';
import { createRoot } from "react-dom/client";
import App from "./app/App.tsx";
import { AuthProvider, setSession } from "./lib/auth.tsx";
import "./styles/index.css";

// Убеждаемся, что React загружен
console.log('React version:', React.version);

/**
 * Вход с хаба ЮФУ: hub.sfedu.ru открывает наш сайт с ?token=<JWT>.
 * Забираем токен из URL, сразу чистим адрес (чтобы не светился в истории/логах),
 * обмениваем его на наш токен через /api/auth/sso и сохраняем сессию.
 */
async function performHubSso(): Promise<void> {
  const params = new URLSearchParams(window.location.search);
  const hubToken = params.get("token");
  if (!hubToken) return;

  params.delete("token");
  const query = params.toString();
  const cleanUrl = window.location.pathname + (query ? `?${query}` : "") + window.location.hash;
  window.history.replaceState({}, document.title, cleanUrl);

  try {
    const res = await fetch("/api/auth/sso", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: hubToken }),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) return;
    const data = await res.json();
    setSession({ token: data.token, role: data.role, name: data.name, username: data.username });
  } catch {
    // если сервис ЮФУ недоступен — просто покажем обычный экран входа
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
      <AuthProvider>
        <App />
      </AuthProvider>
    </React.StrictMode>
  );
}

bootstrap();
