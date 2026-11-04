import { useState } from "react";
import { ArrowRight, GraduationCap, LogIn, ShieldCheck, Sparkles, TrendingUp } from "lucide-react";

import { ApiError, request } from "../../lib/api";
import { HUB_LOGIN_URL } from "../../lib/hubToken";
import { Button, Field, inputClass } from "../../design/primitives";

const FEATURES = [
  {
    icon: TrendingUp,
    title: "Данные рынка труда",
    text: "Вакансии hh.ru собираются автоматически и обновляются по расписанию.",
  },
  {
    icon: Sparkles,
    title: "Gap-анализ программ",
    text: "Сравнение учебного плана с требованиями работодателей по каждой компетенции.",
  },
  {
    icon: GraduationCap,
    title: "Отчёты и прогнозы",
    text: "Готовые отчёты для аккредитации и прогноз спроса на навыки.",
  },
];

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) {
      // Сервер сам объясняет, когда разблокируется аккаунт.
      return err.message || "Слишком много попыток входа. Попробуйте позже.";
    }
    if (err.status >= 500) return "Сервер временно недоступен. Попробуйте позже.";
    if (err.status === 401) return "Неверный логин или пароль";
    return err.message || "Не удалось войти";
  }
  return "Нет связи с сервером. Проверьте подключение.";
}

export function LoginPage({
  onLogin,
}: {
  onLogin: (token: string, role: string, name: string) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  // Причина неудачного входа через хаб приходит из main.tsx через
  // sessionStorage: без этого ошибка видна только в консоли.
  const [ssoError] = useState(() => {
    try {
      const stored = sessionStorage.getItem("ssoError");
      if (stored) sessionStorage.removeItem("ssoError");
      return stored ?? "";
    } catch {
      return "";
    }
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await request<{ token: string; role: string; name: string }>(
        "/api/auth/login",
        { method: "POST", body: { email: username, password }, anonymous: true },
      );
      onLogin(data.token, data.role, data.name);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.1fr_1fr] bg-background">
      {/* Brand panel */}
      <aside className="relative hidden lg:flex flex-col justify-between overflow-hidden bg-gradient-to-br from-primary via-blue-600 to-sky-500 p-12 text-white">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-30"
          style={{
            backgroundImage:
              "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.35) 0, transparent 45%), radial-gradient(circle at 80% 70%, rgba(255,255,255,0.25) 0, transparent 40%)",
          }}
        />
        <div className="relative">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-xl bg-white/15 ring-1 ring-white/30 backdrop-blur">
              <TrendingUp className="size-6" />
            </div>
            <div>
              <div className="text-base font-semibold tracking-tight">Карта компетенций</div>
              <div className="text-xs text-white/70">Competency Gap Analyzer</div>
            </div>
          </div>

          <h2 className="mt-14 max-w-md text-3xl font-semibold leading-tight tracking-tight">
            Соединяем учебные программы с реальными требованиями рынка
          </h2>
          <p className="mt-3 max-w-md text-sm text-white/80">
            Аналитика компетенций, gap-анализ и прогнозы — в одном рабочем месте
            для преподавателей, руководителей программ и студентов.
          </p>

          <ul className="mt-10 space-y-5">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3">
                <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-white/15 ring-1 ring-white/25">
                  <Icon className="size-4" />
                </div>
                <div>
                  <div className="text-sm font-medium">{title}</div>
                  <div className="text-xs text-white/75">{text}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="relative flex items-center gap-2 text-xs text-white/70">
          <ShieldCheck className="size-4" />
          Персональные данные обрабатываются только внутри системы
        </div>
      </aside>

      {/* Form panel */}
      <main className="flex items-center justify-center p-6 sm:p-10">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <div className="flex size-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-sky-500 text-white shadow-md shadow-primary/25">
              <TrendingUp className="size-5" />
            </div>
            <div>
              <div className="text-sm font-semibold text-foreground">Карта компетенций</div>
              <div className="text-[11px] text-muted-foreground">Competency Gap Analyzer</div>
            </div>
          </div>

          <h1 className="text-2xl font-semibold tracking-tight text-foreground">Вход в систему</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            Рабочая почта и пароль или учётная запись ЮФУ.
          </p>

          <a
            href={HUB_LOGIN_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-7 flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 text-sm font-medium text-foreground transition-colors hover:bg-muted active:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <LogIn className="size-4" />
            Войти через hub.sfedu.ru
          </a>
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
            Нужен для публикаций и статей по программам: хаб передаёт токен доступа,
            после входа вы вернётесь сюда автоматически.
          </p>

          <div className="mt-6 flex items-center gap-3 text-[11px] uppercase tracking-wide text-muted-foreground">
            <span className="h-px flex-1 bg-border" />
            или локальный вход
            <span className="h-px flex-1 bg-border" />
          </div>

          {ssoError && (
            <div
              role="alert"
              className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
            >
              Вход через хаб не удался: {ssoError}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-4 space-y-4" noValidate>
            <Field label="Почта" htmlFor="username">
              <input
                id="username"
                type="email"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="name@university.ru"
                required
                className={inputClass}
              />
            </Field>

            <Field label="Пароль" htmlFor="password">
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className={inputClass}
              />
            </Field>

            {error && (
              <div
                role="alert"
                className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
              >
                {error}
              </div>
            )}

            <Button type="submit" loading={loading} className="w-full" iconRight={<ArrowRight className="size-4" />}>
              {loading ? "Вход…" : "Войти"}
            </Button>
          </form>

          <p className="mt-6 text-[11px] leading-relaxed text-muted-foreground">
            Забыли пароль? Обратитесь к администратору системы — самостоятельный сброс
            пока не предусмотрен.
          </p>
        </div>
      </main>
    </div>
  );
}
