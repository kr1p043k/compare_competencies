import { useCallback, useEffect, useState } from "react";

import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";
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
} from "../../design";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

/** Ответ GET /api/auth/2fa/status. */
interface TwoFactorStatus {
  globally_enabled: boolean;
  configured: boolean;
  confirmed: boolean;
}

/** Ответ POST /api/auth/2fa/setup. */
interface TwoFactorSetup {
  otpauth_uri: string;
  secret: string;
}

/** Ответ POST /api/auth/2fa/confirm (recovery-коды отдаются один раз). */
interface TwoFactorConfirmResponse {
  status?: string;
  recovery_codes?: string[];
}

const STATUS_PATH = "/api/auth/2fa/status";
const SETUP_PATH = "/api/auth/2fa/setup";
const CONFIRM_PATH = "/api/auth/2fa/confirm";
const DISABLE_PATH = "/api/auth/2fa/disable";

/** TOTP-код - ровно шесть цифр (клиентская валидация перед POST /confirm). */
const TOTP_CODE_RE = /^\d{6}$/;
/** Минимальная длина кода отключения: TOTP или длинный recovery-код. */
const MIN_DISABLE_CODE = 4;

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

/**
 * Личный блок «Двухфакторная аутентификация» на странице профиля.
 *
 * Состояния по GET /api/auth/2fa/status:
 * - флаг выключен глобально -> «выключена администратором», кнопка неактивна;
 * - не настроена         -> «Включить 2FA» (setup -> confirm -> recovery-коды);
 * - подтверждена         -> «включена» + «Отключить» с проверкой кода.
 */
export function TwoFactorCard() {
  const [status, setStatus] = useState<TwoFactorStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Диалог настройки.
  const [setupOpen, setSetupOpen] = useState(false);
  const [setup, setSetup] = useState<TwoFactorSetup | null>(null);
  const [setupBusy, setSetupBusy] = useState(false);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [recovery, setRecovery] = useState<string[] | null>(null);

  // Диалог отключения.
  const [disableOpen, setDisableOpen] = useState(false);
  const [disableCode, setDisableCode] = useState("");
  const [disableBusy, setDisableBusy] = useState(false);
  const [disableError, setDisableError] = useState<string | null>(null);

  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const s = await request<TwoFactorStatus>(STATUS_PATH);
      setStatus(s);
    } catch (e) {
      reportError("TwoFactorCard:status", e);
      setLoadError(e instanceof Error ? e.message : "Не удалось загрузить состояние 2FA");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const startSetup = async () => {
    if (status && !status.globally_enabled) return;
    setSetupOpen(true);
    setSetup(null);
    setSetupError(null);
    setCode("");
    setRecovery(null);
    setConfirmError(null);
    setSetupBusy(true);
    try {
      const d = await request<TwoFactorSetup>(SETUP_PATH, { method: "POST" });
      setSetup(d);
    } catch (e) {
      setSetupError(e instanceof Error ? e.message : "Не удалось начать настройку 2FA");
    } finally {
      setSetupBusy(false);
    }
  };

  const confirmCode = async () => {
    if (!TOTP_CODE_RE.test(code) || confirmBusy) return;
    setConfirmBusy(true);
    setConfirmError(null);
    setCopiedKey(null);
    try {
      const d = await request<TwoFactorConfirmResponse>(CONFIRM_PATH, {
        method: "POST",
        body: { code },
      });
      setRecovery(Array.isArray(d.recovery_codes) ? d.recovery_codes : []);
    } catch (e) {
      setConfirmError(e instanceof Error ? e.message : "Не удалось подтвердить код");
    } finally {
      setConfirmBusy(false);
    }
  };

  const closeSetupDialog = () => {
    if (recovery) {
      // Пользователь закрыл диалог, не нажав «Сохранил», - коды показаны один раз.
      setNote("Recovery-коды показаны только один раз. Если не сохранили их - отключите 2FA и настройте заново.");
    }
    setSetupOpen(false);
    setSetup(null);
    setRecovery(null);
    setCode("");
    setConfirmError(null);
    setSetupError(null);
    setCopiedKey(null);
  };

  const acknowledgeRecovery = () => {
    setSetupOpen(false);
    setSetup(null);
    setRecovery(null);
    setCode("");
    setConfirmError(null);
    setCopiedKey(null);
    setNote(null);
    void load();
  };

  const disableTwoFactor = async () => {
    if (disableCode.trim().length < MIN_DISABLE_CODE || disableBusy) return;
    setDisableBusy(true);
    setDisableError(null);
    try {
      await request<{ status?: string }>(DISABLE_PATH, {
        method: "POST",
        body: { code: disableCode.trim() },
      });
      setDisableOpen(false);
      setDisableCode("");
      void load();
    } catch (e) {
      setDisableError(e instanceof Error ? e.message : "Не удалось отключить 2FA");
    } finally {
      setDisableBusy(false);
    }
  };

  const handleCopy = async (key: string, text: string) => {
    const ok = await copyText(text);
    if (ok) {
      setCopiedKey(key);
      setNote(null);
    } else {
      setCopiedKey(null);
      setNote("Не удалось скопировать - выделите текст и скопируйте вручную");
    }
  };

  const CopyHeader = ({
    copyKey,
    title,
    text,
  }: {
    copyKey: string;
    title: string;
    text: string;
  }) => (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs font-medium text-foreground">{title}</span>
      <Button
        size="sm"
        variant="secondary"
        onClick={() => void handleCopy(copyKey, text)}
        aria-label={`Скопировать ${title}`}
      >
        {copiedKey === copyKey ? "Скопировано" : "Скопировать"}
      </Button>
    </div>
  );

  return (
    <Card id="profile-2fa">
      <CardHeader
        title="Двухфакторная аутентификация"
        description="Второй фактор при входе: одноразовый код из приложения-аутентификатора"
      />
      <CardBody className="space-y-3">
        {loading && (
          <div className="space-y-2" aria-label="Загрузка состояния 2FA">
            <Skeleton className="h-6 w-56" />
            <Skeleton className="h-4 w-72" />
          </div>
        )}
        {loadError && !loading && <ErrorState title="Не удалось загрузить состояние 2FA" description={loadError} onRetry={() => void load()} />}
        {status && !loading && (
          <>
            {status.globally_enabled ? (
              status.confirmed ? (
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="success">включена</Badge>
                    <span className="text-xs text-muted-foreground">
                      При входе запрашивается код из приложения
                    </span>
                  </div>
                  <Button variant="secondary" onClick={() => setDisableOpen(true)}>
                    Отключить
                  </Button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="neutral">выключена</Badge>
                    <span className="text-xs text-muted-foreground">
                      {status.configured
                        ? "Настройка не завершена: подтвердите код из приложения"
                        : "Вход только по паролю"}
                    </span>
                  </div>
                  <Button onClick={() => void startSetup()}>Включить 2FA</Button>
                </div>
              )
            ) : (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="neutral">выключена администратором</Badge>
                  <span className="text-xs text-muted-foreground">
                    {status.configured
                      ? "Ваша настройка сохранена и заработает после включения флага"
                      : "Второй фактор отключён на уровне системы"}
                  </span>
                </div>
                <Button disabled>Включить 2FA</Button>
              </div>
            )}
            {note && (
              <p className="text-xs text-muted-foreground" role="status">
                {note}
              </p>
            )}
          </>
        )}
      </CardBody>

      {/* Диалог настройки: секрет + otpauth-URI + подтверждение кодом + recovery-коды. */}
      <Dialog open={setupOpen} onOpenChange={(open) => { if (!open) closeSetupDialog(); }}>
        <DialogContent aria-describedby={undefined}>
          <DialogHeader>
            <DialogTitle>Настройка 2FA</DialogTitle>
            <DialogDescription>
              Введите ключ в приложение-аутентификатор (Google Authenticator, Яндекс.Ключ
              и другие), затем подтвердите код.
            </DialogDescription>
          </DialogHeader>

          {setupBusy && (
            <div className="space-y-2" aria-label="Загрузка ключа 2FA">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          )}
          {setupError && (
            <p className="text-sm text-destructive" role="alert">
              {setupError}
            </p>
          )}

          {recovery !== null && (
            <div className="space-y-3" role="status">
              <p className="text-sm">
                2FA включена. Recovery-коды показываются <strong>один раз</strong> -
                сохраните их, они пригодятся при потере телефона.
              </p>
              <div className="space-y-1">
                {recovery.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    Сервер не вернул recovery-коды.
                  </p>
                ) : (
                  <>
                    <CopyHeader copyKey="codes" title="Recovery-коды" text={recovery.join("\n")} />
                    <ul className="max-h-48 overflow-y-auto rounded-lg border border-border bg-muted/50 p-3">
                      {recovery.map((c) => (
                        <li key={c} className="font-mono text-sm text-foreground">
                          {c}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
              <Button onClick={acknowledgeRecovery}>Сохранил</Button>
            </div>
          )}

          {recovery === null && setup && !setupBusy && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <CopyHeader copyKey="uri" title="Ключ для приложения (otpauth)" text={setup.otpauth_uri} />
                <code className="block break-all rounded-lg border border-border bg-muted/50 px-3 py-2 font-mono text-xs text-foreground">
                  {setup.otpauth_uri}
                </code>
              </div>
              <div className="space-y-1.5">
                <CopyHeader copyKey="secret" title="Секрет" text={setup.secret} />
                <code className="block break-all rounded-lg border border-border bg-muted/50 px-3 py-2 font-mono text-sm text-foreground">
                  {setup.secret}
                </code>
              </div>
              <Field
                label="Код из приложения"
                hint="Шесть цифр, которые приложение показывает прямо сейчас"
                error={confirmError ?? undefined}
                htmlFor="twofa-confirm-code"
              >
                <input
                  id="twofa-confirm-code"
                  className={inputClass}
                  value={code}
                  onChange={(e) => {
                    setCode(e.target.value.replace(/\D/g, "").slice(0, 6));
                    setConfirmError(null);
                  }}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  placeholder="000000"
                  aria-label="Код из приложения"
                />
              </Field>
              <Button
                onClick={() => void confirmCode()}
                disabled={!TOTP_CODE_RE.test(code) || confirmBusy}
                loading={confirmBusy}
                className="w-full"
              >
                Подтвердить
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Диалог отключения: нужен код из приложения или recovery-код. */}
      <Dialog open={disableOpen} onOpenChange={(open) => { if (!open) { setDisableOpen(false); setDisableCode(""); setDisableError(null); } }}>
        <DialogContent aria-describedby={undefined}>
          <DialogHeader>
            <DialogTitle>Отключить 2FA</DialogTitle>
            <DialogDescription>
              Введите код из приложения-аутентификатора или один из recovery-кодов,
              чтобы подтвердить отключение.
            </DialogDescription>
          </DialogHeader>
          <Field
            label="Код из приложения или recovery-код"
            error={disableError ?? undefined}
            htmlFor="twofa-disable-code"
          >
            <input
              id="twofa-disable-code"
              className={inputClass}
              value={disableCode}
              onChange={(e) => {
                setDisableCode(e.target.value);
                setDisableError(null);
              }}
              autoComplete="one-time-code"
              aria-label="Код из приложения или recovery-код"
            />
          </Field>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setDisableOpen(false)}>
              Отмена
            </Button>
            <Button
              variant="danger"
              onClick={() => void disableTwoFactor()}
              disabled={disableCode.trim().length < MIN_DISABLE_CODE || disableBusy}
              loading={disableBusy}
            >
              Отключить 2FA
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
