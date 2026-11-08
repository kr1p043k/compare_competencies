import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright для E2E-смоука (frontend/e2e).
 *
 * По умолчанию Playwright сам НЕ поднимает серверы: прогон рассчитан на уже
 * запущенные backend (:8000) и frontend (:3000) - см. e2e/README.md. Поднять
 * vite руками Playwright можно только явно: E2E_START_WEBSERVER=1 (backend
 * всё равно нужно запускать отдельно, смоук-тесты ходят в /api через прокси
 * vite или напрямую в API_BASE_URL).
 */

const baseURL = process.env.E2E_BASE_URL || "http://localhost:3000";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  // Смоук должен падать громко, а не зашумливаться ретраями; но один ретрай
  // нужен, чтобы trace: on-first-retry реально успел что-то записать.
  retries: process.env.CI ? 2 : 1,
  reporter: "list",
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // Не задано - веб-сервер не стартует вовсе (требование смоука: серверы
  // поднимает окружение, а не тест).
  webServer: process.env.E2E_START_WEBSERVER
    ? {
        command: "npm run dev",
        url: baseURL,
        reuseExistingServer: !process.env.CI,
      }
    : undefined,
});
