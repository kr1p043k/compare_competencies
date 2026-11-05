/**
 * Управление двухфакторной аутентификацией (TOTP) в профиле администратора.
 *
 * Секция для AdminProfilePage: включение (секрет + подтверждение кодом),
 * показ резервных кодов ровно один раз, ротация и отключение через диалог
 * с явным подтверждением словом. Секреты и коды не логируются и не
 * сохраняются в localStorage/sessionStorage - живут только в состоянии
 * компонента до закрытия панели.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Copy, Download, ExternalLink, Info, Loader2, RefreshCw, ShieldAlert, ShieldCheck } from "lucide-react";

import { ApiError, request, saveBlob } from "../../lib/api";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  ErrorState,
  Field,
  Skeleton,
  inputClass,
} from "../../design/primitives";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

/** GET /api/auth/me - нас интересует состояние TOTP. */
type MeResponse = {
  username?: string;
  role?: string;
  name?: string;
  two_factor_enabled?: boolean;
  totp_enabled?: boolean;
  totp_locked_until?: string | null;
};

/** POST /api/auth/2fa/setup. */
type SetupData = { secret: string; otpauth_uri: string };

/** Внутренние панели секции. */
type Panel = "idle" | "setup" | "codes" | "rotate";

function errMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    if (err.status >= 500) return "Сервер временно недоступен. Попробуйте позже.";
    // 401 «Неверный код», 429 блокировка - текст отдаёт сервер.
    return err.message || fallback;
  }
  return "Нет связи с сервером. Проверьте подключение.";
}

/** Копирование с запасным путём для контекстов без Clipboard API. */
async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Clipboard API недоступен - пробуем запасной путь ниже.
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.readOnly = true;
    ta.className = "fixed top-0 h-px w-px opacity-0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

function parseCodes(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((c): c is string => typeof c === "string" && c.length > 0);
}

export function AdminSecurity() {
  // Состояние секции из /api/auth/me.
  const [boot, setBoot] = useState<"loading" | "error" | "ready">("loading");
  const [bootError, setBootError] = useState("");
  const [enabled, setEnabled] = useState(false);
  // Глобальный переключатель из админ-консоли: пока он выключен,
  // личная настройка TOTP недоступна.
  const [globalOn, setGlobalOn] = useState(true);
  const [panel, setPanel] = useState<Panel>("idle");

  // Панель включения: секрет из setup + код подтверждения.
  const [setup, setSetup] = useState<SetupData | null>(null);
  const [setupBusy, setSetupBusy] = useState(false);
  const [setupStartError, setSetupStartError] = useState("");
  const [setupCode, setSetupCode] = useState("");
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [confirmError, setConfirmError] = useState("");
  const [secretNotice, setSecretNotice] = useState("");

  // Одноразовая панель резервных кодов (после confirm или rotate).
  const [codes, setCodes] = useState<string[]>([]);
  const [codesSaved, setCodesSaved] = useState(false);
  const [codesNotice, setCodesNotice] = useState("");
  const [codesNoticeOk, setCodesNoticeOk] = useState(true);

  // Ротация кодов: нужен текущий код.
  const [rotateCode, setRotateCode] = useState("");
  const [rotateBusy, setRotateBusy] = useState(false);
  const [rotateError, setRotateError] = useState("");

  // Отключение: диалог со словом-подтверждением и текущим кодом.
  const [disableOpen, setDisableOpen] = useState(false);
  const [disableWord, setDisableWord] = useState("");
  const [disableCode, setDisableCode] = useState("");
  const [disableBusy, setDisableBusy] = useState(false);
  const [disableError, setDisableError] = useState("");

  const setupCodeRef = useRef<HTMLInputElement>(null);
  const rotateCodeRef = useRef<HTMLInputElement>(null);

  const loadMe = useCallback(async () => {
    setBoot("loading");
    setBootError("");
    try {
      const me = await request<MeResponse>("/api/auth/me");
      setEnabled(me.totp_enabled === true);
      setGlobalOn(me.two_factor_enabled !== false);
      setBoot("ready");
    } catch (err) {
      setBootError(errMessage(err, "Не удалось загрузить настройки безопасности"));
      setBoot("error");
    }
  }, []);

  useEffect(() => {
    void loadMe();
  }, [loadMe]);

  // Фокус в поле кода сразу при открытии панели.
  useEffect(() => {
    if (panel === "setup") setupCodeRef.current?.focus();
    if (panel === "rotate") rotateCodeRef.current?.focus();
  }, [panel]);

  const startSetup = async () => {
    setSetupStartError("");
    setSetupBusy(true);
    try {
      const d = await request<{ otpauth_uri?: string; secret?: string }>("/api/auth/2fa/setup", {
        method: "POST",
      });
      const secret = typeof d?.secret === "string" ? d.secret : "";
      if (!secret) {
        setSetupStartError("Сервер не вернул секрет. Попробуйте ещё раз.");
        return;
      }
      setSetup({ secret, otpauth_uri: typeof d.otpauth_uri === "string" ? d.otpauth_uri : "" });
      setSetupCode("");
      setConfirmError("");
      setSecretNotice("");
      setPanel("setup");
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // Уже включена - показываем фактическое состояние с сервера.
        setSetupStartError("Двухфакторная аутентификация уже включена.");
        void loadMe();
        return;
      }
      setSetupStartError(errMessage(err, "Не удалось начать настройку. Попробуйте позже."));
    } finally {
      setSetupBusy(false);
    }
  };

  const cancelSetup = () => {
    setSetup(null);
    setSetupCode("");
    setConfirmError("");
    setSecretNotice("");
    setPanel("idle");
  };

  const confirmSetup = async () => {
    if (!setup) return;
    if (!/^\d{6}$/.test(setupCode)) {
      setConfirmError("Введите 6 цифр из приложения-аутентификатора.");
      setupCodeRef.current?.focus();
      return;
    }
    setConfirmBusy(true);
    setConfirmError("");
    try {
      const d = await request<{ status?: string; recovery_codes?: unknown }>("/api/auth/2fa/confirm", {
        method: "POST",
        body: { code: setupCode },
      });
      const list = parseCodes(d?.recovery_codes);
      if (list.length === 0) {
        setConfirmError("Сервер не вернул резервные коды. Попробуйте ещё раз.");
        return;
      }
      setSetup(null);
      setSetupCode("");
      setCodes(list);
      setCodesSaved(false);
      setCodesNotice("");
      setEnabled(true);
      setPanel("codes");
    } catch (err) {
      setConfirmError(errMessage(err, "Не удалось подтвердить код."));
    } finally {
      setConfirmBusy(false);
    }
  };

  const rotateCodes = async () => {
    const codeValue = rotateCode.trim();
    if (!codeValue) {
      setRotateError("Введите текущий код из приложения или резервный код.");
      rotateCodeRef.current?.focus();
      return;
    }
    setRotateBusy(true);
    setRotateError("");
    try {
      const d = await request<{ recovery_codes?: unknown }>("/api/auth/2fa/recovery-codes/rotate", {
        method: "POST",
        body: { code: codeValue },
      });
      const list = parseCodes(d?.recovery_codes);
      if (list.length === 0) {
        setRotateError("Сервер не вернул новые коды. Попробуйте ещё раз.");
        return;
      }
      setRotateCode("");
      setCodes(list);
      setCodesSaved(false);
      setCodesNotice("");
      setPanel("codes");
    } catch (err) {
      setRotateError(errMessage(err, "Не удалось обновить коды."));
    } finally {
      setRotateBusy(false);
    }
  };

  const cancelRotate = () => {
    setRotateCode("");
    setRotateError("");
    setPanel("idle");
  };

  const finishCodes = () => {
    setCodes([]);
    setCodesSaved(false);
    setCodesNotice("");
    setPanel("idle");
  };

  const copySecret = async () => {
    if (!setup) return;
    const ok = await copyToClipboard(setup.secret);
    setSecretNotice(ok ? "Секрет скопирован" : "Не удалось скопировать - выделите текст и скопируйте вручную");
  };

  const copyCodes = async () => {
    if (codes.length === 0) return;
    const ok = await copyToClipboard(codes.join("\n"));
    setCodesNoticeOk(ok);
    setCodesNotice(ok ? "Коды скопированы" : "Не удалось скопировать - выделите текст и скопируйте вручную");
  };

  const downloadCodes = () => {
    if (codes.length === 0) return;
    const text = [
      "Резервные коды двухфакторной аутентификации",
      "Каждый код работает один раз.",
      "",
      ...codes,
    ].join("\r\n");
    // saveBlob сам создаёт object URL, скачивает и освобождает его.
    saveBlob(new Blob([text], { type: "text/plain;charset=utf-8" }), "recovery-codes.txt");
  };

  const openDisableDialog = () => {
    setDisableWord("");
    setDisableCode("");
    setDisableError("");
    setDisableOpen(true);
  };

  const wordOk = disableWord.trim() === "ОТКЛЮЧИТЬ";

  const doDisable = async () => {
    const codeValue = disableCode.trim();
    if (!wordOk || !codeValue) return;
    setDisableBusy(true);
    setDisableError("");
    try {
      await request<{ status?: string }>("/api/auth/2fa/disable", {
        method: "POST",
        body: { code: codeValue },
      });
      setDisableOpen(false);
      setDisableWord("");
      setDisableCode("");
      setDisableError("");
      setPanel("idle");
      setSetup(null);
      setSetupCode("");
      setCodes([]);
      await loadMe();
    } catch (err) {
      setDisableError(errMessage(err, "Не удалось отключить двухфакторную аутентификацию."));
    } finally {
      setDisableBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader
        title={
          <span className="flex flex-wrap items-center gap-2">
            <ShieldCheck className="size-5 text-primary" />
            Двухфакторная аутентификация
            {boot === "ready" && (
              <Badge tone={enabled ? "success" : "warning"}>
                {enabled ? "включена" : "выключена"}
              </Badge>
            )}
          </span>
        }
        description="Дополнительный одноразовый код при входе - защита на случай, если пароль узнают другие"
      />
      <CardBody className="space-y-4">
        {boot === "ready" && !globalOn && (
          <div
            role="note"
            className="flex items-start gap-2 rounded-lg border border-info/40 bg-info-soft px-3 py-2 text-xs text-info"
          >
            <Info className="mt-0.5 size-4 shrink-0" />
            <span>
              Глобально выключено администратором. Включить двухфакторную аутентификацию можно на
              вкладке «Безопасность» в консоли администрирования.
            </span>
          </div>
        )}

        {boot === "loading" && (
          <div className="space-y-2" aria-label="Загрузка настроек безопасности">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-10 w-44" />
          </div>
        )}

        {boot === "error" && <ErrorState description={bootError} onRetry={() => void loadMe()} />}

        {boot === "ready" && !enabled && (
          <>
            {panel === "setup" && setup ? (
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Добавьте аккаунт в приложение-аутентификатор (Google Authenticator, Aegis,
                  1Password) и введите этот секрет вручную.
                </p>
                <div className="rounded-lg border border-border bg-muted/50 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <code className="min-w-0 break-all select-all font-mono text-sm text-foreground">
                      {setup.secret}
                    </code>
                    <Button variant="secondary" size="sm" onClick={() => void copySecret()}>
                      <Copy className="size-3.5" />
                      Скопировать
                    </Button>
                  </div>
                  <p role="status" aria-live="polite" className="mt-2 min-h-4 text-[11px] text-muted-foreground">
                    {secretNotice}
                  </p>
                </div>
                {setup.otpauth_uri && (
                  <a
                    href={setup.otpauth_uri}
                    className="inline-flex items-center gap-1.5 break-all text-sm font-medium text-primary underline underline-offset-4 hover:opacity-80"
                  >
                    <ExternalLink className="size-4 shrink-0" />
                    Открыть в приложении-аутентификаторе
                  </a>
                )}
                <form
                  className="space-y-3"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void confirmSetup();
                  }}
                  noValidate
                >
                  <Field
                    label="Код из приложения"
                    htmlFor="setup-code"
                    hint="6 цифр, обновляется каждые 30 секунд"
                  >
                    <input
                      ref={setupCodeRef}
                      id="setup-code"
                      type="text"
                      className={inputClass}
                      value={setupCode}
                      onChange={(e) => setSetupCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
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
                  {confirmError && (
                    <p role="alert" className="text-xs text-destructive">
                      {confirmError}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <Button type="submit" disabled={confirmBusy} aria-busy={confirmBusy || undefined}>
                      {confirmBusy ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Check className="size-4" />
                      )}
                      Подтвердить
                    </Button>
                    <Button type="button" variant="secondary" onClick={cancelSetup} disabled={confirmBusy}>
                      Отмена
                    </Button>
                  </div>
                </form>
              </div>
            ) : (
              <>
                <p className="text-sm text-muted-foreground">
                  Двухфакторная аутентификация выключена. Включите её, чтобы при каждом входе
                  запрашивался одноразовый код из приложения-аутентификатора.
                </p>
                {setupStartError && (
                  <p role="alert" className="text-xs text-destructive">
                    {setupStartError}
                  </p>
                )}
                <Button
                  onClick={() => void startSetup()}
                  disabled={setupBusy || !globalOn}
                  aria-busy={setupBusy || undefined}
                  title={
                    !globalOn
                      ? "Доступно после включения двухфакторной аутентификации на вкладке «Безопасность»"
                      : undefined
                  }
                >
                  {setupBusy ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <ShieldCheck className="size-4" />
                  )}
                  Включить
                </Button>
              </>
            )}
          </>
        )}

        {boot === "ready" && enabled && (
          <>
            {panel === "codes" && codes.length > 0 ? (
              <div className="space-y-3">
                <p className="flex items-start gap-2 rounded-lg border border-warning/40 bg-warning-soft px-3 py-2 text-xs text-warning">
                  <ShieldAlert className="mt-0.5 size-4 shrink-0" />
                  <span>
                    Резервные коды показываются только один раз. Сохраните их в надёжном месте: это
                    запасной способ входа без телефона.
                  </span>
                </p>
                <ul className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                  {codes.map((c) => (
                    <li
                      key={c}
                      className="select-all break-all rounded-md border border-border bg-muted/50 px-2 py-1.5 text-center font-mono text-sm text-foreground"
                    >
                      {c}
                    </li>
                  ))}
                </ul>
                <p
                  role="status"
                  aria-live="polite"
                  className={`min-h-4 text-xs ${codesNoticeOk ? "text-success" : "text-destructive"}`}
                >
                  {codesNotice}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" size="sm" onClick={() => void copyCodes()}>
                    <Copy className="size-3.5" />
                    Скопировать все
                  </Button>
                  <Button variant="secondary" size="sm" onClick={downloadCodes}>
                    <Download className="size-3.5" />
                    Скачать .txt
                  </Button>
                </div>
                <label className="flex cursor-pointer items-start gap-2 text-sm text-foreground">
                  <input
                    type="checkbox"
                    checked={codesSaved}
                    onChange={(e) => setCodesSaved(e.target.checked)}
                    className="mt-0.5 size-4 shrink-0 cursor-pointer accent-primary"
                  />
                  <span>Я сохранил коды и понимаю, что повторно они не будут показаны</span>
                </label>
                <div>
                  <Button size="sm" disabled={!codesSaved} onClick={finishCodes}>
                    Готово
                  </Button>
                </div>
              </div>
            ) : panel === "rotate" ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Введите текущий код из приложения или один из резервных кодов. Прежние резервные
                  коды перестанут работать.
                </p>
                <form
                  className="space-y-3"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void rotateCodes();
                  }}
                  noValidate
                >
                  <Field
                    label="Текущий код"
                    htmlFor="rotate-code"
                    hint="6 цифр из приложения или 8-символьный резервный код"
                  >
                    <input
                      ref={rotateCodeRef}
                      id="rotate-code"
                      type="text"
                      className={inputClass}
                      value={rotateCode}
                      onChange={(e) => setRotateCode(e.target.value)}
                      autoComplete="off"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                      placeholder="Текущий код"
                      required
                    />
                  </Field>
                  {rotateError && (
                    <p role="alert" className="text-xs text-destructive">
                      {rotateError}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <Button type="submit" disabled={rotateBusy} aria-busy={rotateBusy || undefined}>
                      {rotateBusy ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <RefreshCw className="size-4" />
                      )}
                      Сгенерировать новые коды
                    </Button>
                    <Button type="button" variant="secondary" onClick={cancelRotate} disabled={rotateBusy}>
                      Отмена
                    </Button>
                  </div>
                </form>
              </div>
            ) : (
              <>
                <p className="text-sm text-muted-foreground">
                  При входе система дополнительно запрашивает одноразовый код из
                  приложения-аутентификатора.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setRotateCode("");
                      setRotateError("");
                      setPanel("rotate");
                    }}
                  >
                    <RefreshCw className="size-4" />
                    Сменить резервные коды
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={openDisableDialog}
                    className="border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
                  >
                    <ShieldAlert className="size-4" />
                    Отключить
                  </Button>
                </div>
              </>
            )}
          </>
        )}

        <Dialog
          open={disableOpen}
          onOpenChange={(open) => {
            if (!disableBusy) setDisableOpen(open);
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Отключить двухфакторную аутентификацию?</DialogTitle>
              <DialogDescription>
                После отключения для входа потребуется только почта и пароль. Включить защиту снова
                можно будет в любой момент.
              </DialogDescription>
            </DialogHeader>
            <form
              className="space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                void doDisable();
              }}
              noValidate
            >
              <Field
                label="Слово подтверждения"
                htmlFor="disable-word"
                hint="Введите слово ОТКЛЮЧИТЬ"
                error={
                  disableWord.length > 0 && !wordOk ? "Слово не совпадает" : undefined
                }
              >
                <input
                  id="disable-word"
                  value={disableWord}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDisableWord(e.target.value)}
                  autoComplete="off"
                  placeholder="ОТКЛЮЧИТЬ"
                  className={`${inputClass} font-mono`}
                />
              </Field>
              <Field label="Код из приложения или резервный код" htmlFor="disable-code">
                <input
                  id="disable-code"
                  value={disableCode}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDisableCode(e.target.value)}
                  autoComplete="off"
                  placeholder="Текущий код"
                  className={inputClass}
                />
              </Field>
              {disableError && (
                <p role="alert" className="text-xs text-destructive">
                  {disableError}
                </p>
              )}
              <DialogFooter>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => setDisableOpen(false)}
                  disabled={disableBusy}
                >
                  Отмена
                </Button>
                <Button
                  type="submit"
                  variant="danger"
                  disabled={disableBusy || !wordOk || disableCode.trim().length === 0}
                  aria-busy={disableBusy || undefined}
                >
                  {disableBusy && <Loader2 className="size-4 animate-spin" />}
                  Отключить
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardBody>
    </Card>
  );
}
