import { expect, test } from "@playwright/test";

/**
 * E2E smoke: экран логина, отказ при неверных данных, реальный вход и
 * health бэкенда. Каждый тест сам открывает страницу и работает в своём
 * контексте (Playwright изолирует тесты по умолчанию) - межтестовых
 * зависимостей нет.
 *
 * Селекторы взяты из живой разметки, не выдуманы:
 * - label «Почта» <-> input#username, label «Пароль» <-> input#password:
 *   src/app/components/LoginPage.tsx:366-390; сам <label htmlFor> рисует
 *   Field - src/design/primitives.tsx:351-353 (getByLabel работает).
 * - кнопка «Войти»: LoginPage.tsx:401-403, нативный <button>
 *   (src/design/primitives.tsx:57), портит её только loading-вариант
 *   («Вход…»), поэтому клик успевает до переключения.
 * - ошибка входа: div role="alert" (LoginPage.tsx:392-399); текст 401
 *   «Неверный логин или пароль» ставится в LoginPage.tsx:64, сам 401
 *   отдаёт бэкенд: src/api_pkg/routers/auth.py:558.
 * - заголовок login-экрана: h1 «Вход» (LoginPage.tsx:254); заголовок
 *   шелла после входа: h1 «Карта компетенций»
 *   (src/app/shell/AppHeader.tsx:73-75).
 *
 * 2FA: смоук рассчитан на глобальный переключатель 2FA в OFF (это значение
 * по умолчанию, включается в админке) - вход только по паролю. Если рубильник
 * включён, валидный вход админа уйдёт в challenge-экран и тест честно
 * упадёт, а не «пройдёт» мимо смысла.
 *
 * «Остались на логине» проверяем по контенту, а не по URL: приложение
 * hash-роутится и выставляет «#/vacancies» ещё до входа
 * (src/lib/router.ts:40-41 монтируется вместе с App, App.tsx:175), так что
 * URL одинаковый и на логине, и в шелле.
 */

/** База API для health-проверки (FastAPI). */
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

test.describe("smoke: login screen", () => {
  test("login page renders username and password fields with a submit button", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { level: 1, name: "Вход" })).toBeVisible();
    await expect(page.getByLabel("Почта")).toBeVisible();
    await expect(page.getByLabel("Пароль")).toBeVisible();
    await expect(page.getByRole("button", { name: "Войти" })).toBeVisible();
  });

  test("invalid credentials show an error and keep the login screen", async ({ page }) => {
    await page.goto("/");

    // Несуществующий email: 401 из auth.py:558 -> «Неверный логин или пароль»
    // (LoginPage.tsx:64). Фейковый адрес не трогает счётчик неудач реальных
    // аккаунтов (UPDATE по users WHERE email=... не находит строк,
    // auth.py:283-291), но IP-лимит 10/мин (auth.py:501) всё равно бьёт по
    // прогонам - не запускать смоук чаще раза в минуту на одном IP.
    await page.getByLabel("Почта").fill("e2e-smoke@example.com");
    await page.getByLabel("Пароль").fill("not-a-real-password");
    await page.getByRole("button", { name: "Войти" }).click();

    await expect(page.getByRole("alert")).toContainText("Неверный логин или пароль");

    // Экран входа не сменился: заголовок и поля на месте.
    await expect(page.getByRole("heading", { level: 1, name: "Вход" })).toBeVisible();
    await expect(page.getByLabel("Почта")).toBeVisible();
  });
});

test.describe("smoke: real login", () => {
  test("valid credentials land on the app shell", async ({ page }) => {
    const user = process.env.E2E_USER;
    const pass = process.env.E2E_PASS;

    test.skip(
      !user || !pass,
      "E2E_USER / E2E_PASS are not set: skipping the valid-credentials login check",
    );
    if (!user || !pass) return; // защита типов; доехать сюда нельзя - skip выше

    await page.goto("/");

    await page.getByLabel("Почта").fill(user);
    await page.getByLabel("Пароль").fill(pass);
    await page.getByRole("button", { name: "Войти" }).click();

    // Шелл: h1 «Карта компетенций» (AppHeader.tsx:73-75);
    // login-экран размонтирован, h1 «Вход» исчез.
    await expect(page.getByRole("heading", { name: "Карта компетенций" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: "Вход" })).toHaveCount(0);
  });
});

test("health: GET {API_BASE_URL}/api/health returns 200", async ({ request }) => {
  const response = await request.get(`${API_BASE_URL}/api/health`);

  expect(response.status()).toBe(200);
});
