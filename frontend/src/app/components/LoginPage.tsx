import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, GraduationCap, ShieldCheck, Sparkles, TrendingUp } from "lucide-react";

import { ApiError, request } from "../../lib/api";
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

/**
 * Ответ /api/auth/login: либо готовая сессия, либо требование пройти
 * двухфакторное подтверждение (challenge), либо первая настройка 2FA -
 * у админа защита включена, но ключа ещё нет (enroll_id вместо токена).
 * Проверяем форму ответа явно: тип guard-ом, наличие токена - рантайм
 * guard-ом ниже.
 */
type LoginResponse =
  | { token: string; role: string; name: string; username?: string }
  | { status: "totp_required"; challenge_id: string; expires_in?: number }
  | { status: "totp_setup_required"; enroll_id: string; expires_in?: number };

/** Успешный ответ 2FA-шагов (verify/recovery) - только токен, без challenge. */
type TokenBundle = Extract<LoginResponse, { token: string }>;

/**
 * Ответ /api/auth/2fa/setup: секрет показывается ровно один раз.
 * Поля проверяются на месте - серверу не доверяем слепо.
 */
type SetupResponse = { otpauth_uri?: unknown; secret?: unknown; enroll_id?: unknown };

/**
 * Ответ /api/auth/2fa/confirm: recovery-коды (показываются единственный раз).
 * При настройке из экрана входа (по enroll_id) сервер сразу отдаёт сессию,
 * чтобы админ не начинал вход заново.
 */
type EnrollConfirmResponse = {
  status?: string;
  recovery_codes?: unknown;
  token?: unknown;
  role?: unknown;
  name?: unknown;
};

function isChallengeResponse(
  r: LoginResponse,
): r is Extract<LoginResponse, { status: "totp_required" }> {
  return (r as { status?: unknown }).status === "totp_required";
}

function isSetupRequiredResponse(
  r: LoginResponse,
): r is Extract<LoginResponse, { status: "totp_setup_required" }> {
  return (r as { status?: unknown }).status === "totp_setup_required";
}

/** Список recovery-кодов: непустой массив непустых строк (показываются один раз). */
function isRecoveryCodesList(v: unknown): v is string[] {
  return (
    Array.isArray(v) &&
    v.length > 0 &&
    v.every((c) => typeof c === "string" && c.length > 0)
  );
}

/**
 * Рантайм-проверка токена: onLogin нельзя вызывать с пустым token,
 * даже если сервер прислал что-то неожиданное.
 */
function hasValidToken(r: { token?: unknown; role?: unknown; name?: unknown }): r is TokenBundle {
  return (
    typeof r.token === "string" &&
    r.token.length > 0 &&
    typeof r.role === "string" &&
    r.role.length > 0 &&
    typeof r.name === "string"
  );
}

function loginErrorMessage(err: unknown): string {
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

/** Для шага 2FA: 401 здесь значит «неверный код», отдаём текст сервера. */
function codeErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) return err.message || "Слишком много попыток. Попробуйте позже.";
    if (err.status >= 500) return "Сервер временно недоступен. Попробуйте позже.";
    if (err.status === 401) return err.message || "Неверный код";
    return err.message || "Не удалось подтвердить вход";
  }
  return "Нет связи с сервером. Проверьте подключение.";
}

/** Ошибка запроса секрета на шаге первичной настройки. */
function setupErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) return err.message || "Слишком много попыток. Попробуйте позже.";
    if (err.status >= 500) return "Сервер временно недоступен. Попробуйте позже.";
    return err.message || "Не удалось начать настройку 2FA";
  }
  return "Нет связи с сервером. Проверьте подключение.";
}

/** Копирование в буфер: сначала Clipboard API, затем скрытая textarea. */
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* clipboard недоступен - пробуем резервный способ ниже */
  }
  try {
    const helper = document.createElement("textarea");
    helper.value = text;
    helper.setAttribute("readonly", "");
    helper.className = "sr-only";
    document.body.appendChild(helper);
    helper.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(helper);
    return copied;
  } catch {
    return false;
  }
}

const linkClass =
  "rounded text-sm font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";

export function LoginPage({
  onLogin,
}: {
  onLogin: (token: string, role: string, name: string) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // --- Шаг 2FA: сервер вернул challenge_id вместо токена ---
  const [challenge, setChallenge] = useState<{ id: string; expiresIn?: number } | null>(null);
  const [useRecovery, setUseRecovery] = useState(false);
  const [code, setCode] = useState("");
  const codeRef = useRef<HTMLInputElement>(null);

  // --- Шаг настройки 2FA: у админа ещё нет ключа, сервер вернул enroll_id ---
  const [enroll, setEnroll] = useState<{ id: string; expiresIn?: number } | null>(null);
  const [enrollSetup, setEnrollSetup] = useState<{ secret: string; otpauth_uri: string } | null>(
    null,
  );
  // Подтверждение прошло: коды показываются один раз, onLogin - только после кнопки.
  const [enrollDone, setEnrollDone] = useState<{ codes: string[]; bundle: TokenBundle } | null>(
    null,
  );
  const [secretCopied, setSecretCopied] = useState(false);
  const doneHeadingRef = useRef<HTMLHeadingElement>(null);

  // Шаг 2FA открывается (или переключается режим) - фокус в поле кода,
  // чтобы вводить можно было сразу, без лишнего клика.
  useEffect(() => {
    if (challenge || enroll) codeRef.current?.focus();
  }, [challenge, useRecovery, enroll, enrollSetup]);

  // Recovery-коды показаны: поле кода размонтировано, переводим фокус
  // в заголовок, чтобы экранный читалка объявил смену экрана.
  useEffect(() => {
    if (enrollDone) doneHeadingRef.current?.focus();
  }, [enrollDone]);

  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await request<LoginResponse>("/api/auth/login", {
        method: "POST",
        body: { email: username, password },
        anonymous: true,
      });
      if (isChallengeResponse(data)) {
        setChallenge({
          id: data.challenge_id,
          expiresIn: typeof data.expires_in === "number" ? data.expires_in : undefined,
        });
        setCode("");
        setUseRecovery(false);
        return;
      }
      if (isSetupRequiredResponse(data)) {
        setChallenge(null);
        setUseRecovery(false);
        setCode("");
        setEnrollDone(null);
        setEnrollSetup(null);
        setEnroll({
          id: data.enroll_id,
          expiresIn: typeof data.expires_in === "number" ? data.expires_in : undefined,
        });
        // Секрет нужен сразу: без него аккаунт не добавить в приложение.
        await startEnrollSetup(data.enroll_id);
        return;
      }
      if (!hasValidToken(data)) {
        setError("Неожиданный ответ сервера. Попробуйте ещё раз.");
        return;
      }
      onLogin(data.token, data.role, data.name);
    } catch (err) {
      setError(loginErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleCodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!challenge) return;
    setError("");
    if (useRecovery) {
      if (!/^[0-9a-f]{16}$/i.test(code)) {
        setError("Резервный код - 16 символов: цифры и латинские буквы a-f.");
        codeRef.current?.focus();
        return;
      }
    } else if (!/^\d{6}$/.test(code)) {
      setError("Код - 6 цифр из приложения-аутентификатора.");
      codeRef.current?.focus();
      return;
    }
    setLoading(true);
    try {
      const data = await request<TokenBundle>(
        useRecovery ? "/api/auth/2fa/recovery" : "/api/auth/2fa/verify",
        {
          method: "POST",
          body: { challenge_id: challenge.id, code },
          anonymous: true,
        },
      );
      // Явный guard: onLogin вызывается только с непустым токеном.
      if (!hasValidToken(data)) {
        setError("Неожиданный ответ сервера. Попробуйте ещё раз.");
        return;
      }
      onLogin(data.token, data.role, data.name);
    } catch (err) {
      setError(codeErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  /**
   * Запросить секрет настройки по пропуску enroll_id: действует ограниченное
   * время, сервер поглощает его в /2fa/confirm. До подтверждения setup можно
   * повторять (например, после сетевой ошибки).
   */
  const startEnrollSetup = async (enrollId: string) => {
    setError("");
    setSecretCopied(false);
    setEnrollSetup(null);
    setLoading(true);
    try {
      const data = await request<SetupResponse>("/api/auth/2fa/setup", {
        method: "POST",
        body: { enroll_id: enrollId },
        anonymous: true,
      });
      if (
        typeof data.secret !== "string" ||
        data.secret.length === 0 ||
        typeof data.otpauth_uri !== "string" ||
        data.otpauth_uri.length === 0
      ) {
        setError("Неожиданный ответ сервера. Попробуйте ещё раз.");
        return;
      }
      setEnrollSetup({ secret: data.secret, otpauth_uri: data.otpauth_uri });
    } catch (err) {
      setError(setupErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  /** Подтверждение кода на шаге настройки: коды показываем ДО onLogin. */
  const handleEnrollCodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enroll || !enrollSetup) return;
    setError("");
    if (!/^\d{6}$/.test(code)) {
      setError("Код - 6 цифр из приложения-аутентификатора.");
      codeRef.current?.focus();
      return;
    }
    setLoading(true);
    try {
      const data = await request<EnrollConfirmResponse>("/api/auth/2fa/confirm", {
        method: "POST",
        body: { code, enroll_id: enroll.id },
        anonymous: true,
      });
      // Жёсткий guard: сессия и коды обязаны быть в ответе, иначе не идём дальше.
      // (Коды вытаскиваем в отдельную константу - иначе narrowing по
      // hasValidToken съедает поле recovery_codes.)
      const { recovery_codes: enrollCodes } = data;
      if (!isRecoveryCodesList(enrollCodes) || !hasValidToken(data)) {
        setError("Неожиданный ответ сервера. Попробуйте ещё раз.");
        return;
      }
      // onLogin НЕ вызываем сразу: recovery-коды сервер показывает один раз,
      // пользователь должен увидеть их до перехода в приложение.
      setEnrollDone({ codes: enrollCodes, bundle: data });
    } catch (err) {
      setError(codeErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const backToCredentials = () => {
    setChallenge(null);
    setUseRecovery(false);
    setCode("");
    setError("");
    setEnroll(null);
    setEnrollSetup(null);
    setEnrollDone(null);
  };

  const minutesLeft = (() => {
    if (challenge?.expiresIn && challenge.expiresIn > 0) {
      return Math.max(1, Math.round(challenge.expiresIn / 60));
    }
    return 5;
  })();

  // Пропуск на настройку живёт дольше challenge (по умолчанию 10 минут).
  const enrollMinutesLeft = (() => {
    if (enroll?.expiresIn && enroll.expiresIn > 0) {
      return Math.max(1, Math.round(enroll.expiresIn / 60));
    }
    return 10;
  })();

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.1fr_1fr] bg-background">
      {/* Brand panel */}
      <aside className="relative hidden lg:flex flex-col justify-between overflow-hidden bg-gradient-to-br from-primary via-primary to-chart-2 p-12 text-primary-foreground">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_20%_20%,color-mix(in_srgb,var(--primary-foreground)_35%,transparent)_0,transparent_45%),radial-gradient(circle_at_80%_70%,color-mix(in_srgb,var(--primary-foreground)_25%,transparent)_0,transparent_40%)]"
        />
        <div className="relative">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-xl bg-primary-foreground/15 ring-1 ring-primary-foreground/30 backdrop-blur">
              <TrendingUp className="size-6" />
            </div>
            <div>
              <div className="text-base font-semibold tracking-tight">Карта компетенций</div>
              <div className="text-xs text-primary-foreground/70">Competency Gap Analyzer</div>
            </div>
          </div>

          <h2 className="mt-14 max-w-md text-3xl font-semibold leading-tight tracking-tight">
            Соединяем учебные программы с реальными требованиями рынка
          </h2>
          <p className="mt-3 max-w-md text-sm text-primary-foreground/80">
            Аналитика компетенций, gap-анализ и прогнозы - в одном рабочем месте
            для преподавателей, руководителей программ и студентов.
          </p>

          <ul className="mt-10 space-y-5">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3">
                <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary-foreground/15 ring-1 ring-primary-foreground/25">
                  <Icon className="size-4" />
                </div>
                <div>
                  <div className="text-sm font-medium">{title}</div>
                  <div className="text-xs text-primary-foreground/75">{text}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="relative flex items-center gap-2 text-xs text-primary-foreground/70">
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

          <h1 className="text-2xl font-semibold tracking-tight text-foreground">Вход</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            Рабочая почта и пароль или учётная запись ЮФУ.
          </p>

          {challenge ? (
            <section className="mt-7" aria-label="Двухфакторное подтверждение входа">
              <h2 className="text-lg font-semibold tracking-tight text-foreground">
                Подтверждение входа
              </h2>
              <p className="mt-1.5 text-sm text-muted-foreground">
                {useRecovery
                  ? "Введите резервный код - 16 символов."
                  : "Введите 6-значный код из приложения-аутентификатора."}
              </p>

              <form onSubmit={handleCodeSubmit} className="mt-4 space-y-4" noValidate>
                {useRecovery ? (
                  <Field
                    label="Резервный код"
                    htmlFor="recovery-code"
                    hint="Каждый резервный код работает один раз."
                  >
                    <input
                      ref={codeRef}
                      id="recovery-code"
                      type="text"
                      className={inputClass}
                      value={code}
                      onChange={(e) =>
                        setCode(e.target.value.replace(/[^0-9a-fA-F]/g, "").slice(0, 16))
                      }
                      inputMode="text"
                      autoComplete="off"
                      autoCapitalize="characters"
                      autoCorrect="off"
                      spellCheck={false}
                      maxLength={16}
                      placeholder="XXXXXXXXXXXXXXXX"
                      required
                    />
                  </Field>
                ) : (
                  <Field
                    label="Код из приложения"
                    htmlFor="totp-code"
                    hint="Обновляется каждые 30 секунд."
                  >
                    <input
                      ref={codeRef}
                      id="totp-code"
                      type="text"
                      className={inputClass}
                      value={code}
                      onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                      inputMode="numeric"
                      autoComplete="one-time-code"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                      pattern="[0-9]*"
                      maxLength={6}
                      placeholder="000000"
                      required
                    />
                  </Field>
                )}

                {error && (
                  <div
                    role="alert"
                    className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
                  >
                    {error}
                  </div>
                )}

                <Button type="submit" loading={loading} className="w-full" iconRight={<ArrowRight className="size-4" />}>
                  {loading ? "Проверяем…" : "Подтвердить"}
                </Button>
              </form>

              <div className="mt-3 flex items-center justify-between gap-3">
                <button type="button" className={linkClass} onClick={backToCredentials}>
                  <span className="inline-flex items-center gap-1.5">
                    <ArrowLeft className="size-4" />
                    Назад
                  </span>
                </button>
                <button
                  type="button"
                  className={linkClass}
                  onClick={() => {
                    setUseRecovery((v) => !v);
                    setCode("");
                    setError("");
                  }}
                >
                  {useRecovery ? "Войти по коду из приложения" : "Использовать резервный код"}
                </button>
              </div>

              <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
                Запрос на вход действует {minutesLeft}{" "}
                {minutesLeft === 1 ? "минуту" : minutesLeft < 5 ? "минуты" : "минут"}. Если время
                истекло, вернитесь и войдите заново.
              </p>
            </section>
          ) : enroll ? (
            <section className="mt-7" aria-label="Настройка двухфакторной защиты">
              {enrollDone ? (
                <>
                  <h2
                    ref={doneHeadingRef}
                    tabIndex={-1}
                    className="text-lg font-semibold tracking-tight text-foreground"
                  >
                    Двухфакторная защита включена
                  </h2>
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    Сохраните резервные коды: они показываются только один раз и помогут
                    войти, если приложение станет недоступно.
                  </p>
                  <ul
                    className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2"
                    aria-label="Резервные коды"
                  >
                    {enrollDone.codes.map((recoveryCode) => (
                      <li key={recoveryCode}>
                        <code className="block break-all rounded-lg border border-border bg-muted/50 px-3 py-2 font-mono text-sm text-foreground">
                          {recoveryCode}
                        </code>
                      </li>
                    ))}
                  </ul>
                  <Button
                    type="button"
                    className="mt-4 w-full"
                    iconRight={<ArrowRight className="size-4" />}
                    onClick={() =>
                      onLogin(
                        enrollDone.bundle.token,
                        enrollDone.bundle.role,
                        enrollDone.bundle.name,
                      )
                    }
                  >
                    Я сохранил коды, продолжить
                  </Button>
                </>
              ) : (
                <>
                  <h2 className="text-lg font-semibold tracking-tight text-foreground">
                    Настройка двухфакторной защиты
                  </h2>
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    Администратор включил двухфакторную защиту. Добавьте аккаунт в
                    приложение-аутентификатор и введите код из него, чтобы завершить вход.
                  </p>

                  {enrollSetup ? (
                    <div className="mt-4 rounded-lg border border-border bg-card p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-xs font-medium text-foreground">
                            Секрет для ручного добавления
                          </div>
                          <div className="text-[11px] text-muted-foreground">
                            Введите его в приложении вручную, если открыть ссылку не удалось.
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="secondary"
                          size="sm"
                          onClick={() => {
                            if (enrollSetup) void copyText(enrollSetup.secret).then(setSecretCopied);
                          }}
                        >
                          {secretCopied ? "Скопировано" : "Скопировать"}
                        </Button>
                      </div>
                      <code className="mt-2.5 block break-all rounded-md bg-muted/60 px-3 py-2.5 font-mono text-sm tracking-wide text-foreground">
                        {enrollSetup.secret}
                      </code>
                      <a
                        href={enrollSetup.otpauth_uri}
                        className="mt-2.5 inline-flex items-center gap-1.5 text-[11px] font-medium text-primary underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                      >
                        Открыть ссылку otpauth://
                      </a>
                    </div>
                  ) : loading ? (
                    <p role="status" className="mt-4 text-sm text-muted-foreground">
                      Получаем секрет с сервера…
                    </p>
                  ) : null}

                  {error && (
                    <div
                      role="alert"
                      className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
                    >
                      {error}
                    </div>
                  )}

                  {enrollSetup ? (
                    <form
                      onSubmit={handleEnrollCodeSubmit}
                      className="mt-4 space-y-4"
                      noValidate
                    >
                      <Field
                        label="Код из приложения"
                        htmlFor="enroll-code"
                        hint="Обновляется каждые 30 секунд."
                      >
                        <input
                          ref={codeRef}
                          id="enroll-code"
                          type="text"
                          className={inputClass}
                          value={code}
                          onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                          inputMode="numeric"
                          autoComplete="one-time-code"
                          autoCapitalize="none"
                          autoCorrect="off"
                          spellCheck={false}
                          pattern="[0-9]*"
                          maxLength={6}
                          placeholder="000000"
                          required
                        />
                      </Field>
                      <Button
                        type="submit"
                        loading={loading}
                        className="w-full"
                        iconRight={<ArrowRight className="size-4" />}
                      >
                        {loading ? "Проверяем…" : "Подтвердить"}
                      </Button>
                    </form>
                  ) : !loading && error ? (
                    <Button
                      type="button"
                      variant="secondary"
                      className="mt-4 w-full"
                      onClick={() => {
                        if (enroll) void startEnrollSetup(enroll.id);
                      }}
                    >
                      Повторить
                    </Button>
                  ) : null}

                  <div className="mt-3">
                    <button type="button" className={linkClass} onClick={backToCredentials}>
                      <span className="inline-flex items-center gap-1.5">
                        <ArrowLeft className="size-4" />
                        Назад
                      </span>
                    </button>
                  </div>

                  <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
                    Пропуск на настройку действует {enrollMinutesLeft}{" "}
                    {enrollMinutesLeft === 1 ? "минуту" : enrollMinutesLeft < 5 ? "минуты" : "минут"}.
                    Если время истекло, вернитесь и войдите заново.
                  </p>
                </>
              )}
            </section>
          ) : (
            <>

              <form onSubmit={handleCredentialsSubmit} className="mt-4 space-y-4" noValidate>
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

              <p className="mt-6 text-[11px] leading-relaxed text-muted-foreground">Забыли пароль? Обратитесь к администратору системы.</p>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
