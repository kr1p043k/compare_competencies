import { useEffect, useState } from "react";
import { AlertTriangle, KeyRound, ShieldCheck, ShieldOff } from "lucide-react";

import { ApiError, request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import { Badge, Button, Callout, Card, CardBody, CardHeader, inputClass } from "../../design";

interface SwitchState {
  enabled: boolean;
  /** false = хранилище флага не читается, `enabled` - значение по умолчанию. */
  read_ok?: boolean;
  enrolled_admins: number;
}

interface MyState {
  enabled: boolean;
  enrolled: boolean;
  available: boolean;
}

interface SetupResponse {
  otpauth_uri: string;
  secret: string;
}

function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.message || "Не удалось выполнить запрос";
  return "Нет связи с сервером";
}

/**
 * Безопасность админа: глобальный тумблер 2FA и личное подключение TOTP.
 *
 * Мастер-переключатель по умолчанию выключен и хранится в БД: выключение
 * действует сразу для всех процессов. Подключения админов при выключении
 * не удаляются - при следующем включении код снова спрашивается.
 *
 * GET /api/admin/security/2fa/settings отдаёт read_ok: false, если хранилище
 * флага недоступно для чтения - тогда показанное состояние это значение
 * по умолчанию (выключено), а не прочитанное из БД.
 */
export function AdminSecurity() {
  const [state, setState] = useState<SwitchState | null>(null);
  const [me, setMe] = useState<MyState | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [secret, setSecret] = useState<SetupResponse | null>(null);
  const [code, setCode] = useState("");
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);
  const [disableMode, setDisableMode] = useState(false);

  const load = async () => {
    try {
      const [sw, own] = await Promise.all([
        request<SwitchState>("/api/admin/security/2fa/settings"),
        request<MyState>("/api/auth/2fa/status"),
      ]);
      setState(sw);
      setMe(own);
    } catch (err) {
      reportError("AdminSecurity:load", err);
      setMessage(errorText(err));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const toggle = async () => {
    if (!state) return;
    setBusy(true);
    setMessage("");
    try {
      const next = await request<SwitchState>("/api/admin/security/2fa/settings", {
        method: "PUT",
        body: { enabled: !state.enabled },
      });
      // PUT может не вернуть read_ok - сохраняем последнее прочитанное значение.
      setState((prev) => (prev ? { ...prev, ...next } : next));
      setMessage(next.enabled ? "Двухфакторная аутентификация включена" : "Двухфакторная аутентификация выключена");
      await load();
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const startSetup = async () => {
    setBusy(true);
    setMessage("");
    try {
      const data = await request<SetupResponse>("/api/auth/2fa/setup", {
        method: "POST",
        body: {},
      });
      setSecret(data);
      setDisableMode(false);
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const confirmSetup = async () => {
    setBusy(true);
    setMessage("");
    try {
      const data = await request<{ recovery_codes?: string[] }>("/api/auth/2fa/confirm", {
        method: "POST",
        body: { code },
      });
      setRecoveryCodes(data.recovery_codes ?? []);
      setSecret(null);
      setCode("");
      setMessage("2FA подключена");
      await load();
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const disableMine = async () => {
    setBusy(true);
    setMessage("");
    try {
      await request("/api/auth/2fa/disable", { method: "POST", body: { code } });
      setDisableMode(false);
      setCode("");
      setMessage("2FA отключена для вашей учётной записи");
      await load();
    } catch (err) {
      setMessage(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <ShieldCheck className="size-4" aria-hidden="true" />
            Двухфакторная аутентификация
          </span>
        }
      />
      <CardBody className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          {state?.enabled ? (
            <Badge tone="success">Включена</Badge>
          ) : (
            <Badge tone="neutral">Выключена</Badge>
          )}
          <Button
            onClick={toggle}
            disabled={busy || !state}
            variant={state?.enabled ? "danger" : "primary"}
            iconLeft={state?.enabled ? <ShieldOff className="size-4" /> : <ShieldCheck className="size-4" />}
          >
            {state?.enabled ? "Выключить" : "Включить"}
          </Button>
          <span className="text-xs text-muted-foreground">
            подключено админов: {state?.enrolled_admins ?? 0}
          </span>
        </div>

        {state?.read_ok === false && (
          <Callout
            role="alert"
            tone="warning"
            className="text-xs"
            icon={<AlertTriangle className="size-4" aria-hidden="true" />}
          >
            состояние не читается: действует значение по умолчанию (выключено)
          </Callout>
        )}

        <p className="text-xs text-muted-foreground">
          Выключено - вход по паролю для всех. Включено - код спрашивается у админов,
          которые подключили TOTP; остальные при входе проходят настройку.
          Выключение не удаляет подключения.
        </p>

        <div className="border-t border-border pt-4 space-y-3">
          <div className="flex items-center gap-2 text-sm">
            <KeyRound className="size-4" aria-hidden="true" />
            Ваша учётная запись:{" "}
            {me?.enrolled ? (
              <Badge tone="success">защищена</Badge>
            ) : (
              <Badge tone="neutral">без кода</Badge>
            )}
          </div>

          {!me?.enrolled && !secret && (
            <Button variant="secondary" onClick={startSetup} disabled={busy || !state?.enabled}>
              Подключить приложение-аутентификатор
            </Button>
          )}
          {!state?.enabled && (
            <p className="text-xs text-muted-foreground">
              Сначала включите 2FA тумблером выше.
            </p>
          )}

          {secret && (
            <div className="space-y-3 rounded-lg border border-border bg-muted/40 p-3">
              <p className="text-xs text-muted-foreground">
                Добавьте аккаунт в приложение вручную по ключу, затем введите код.
              </p>
              <div className="break-all rounded bg-surface px-2 py-1 font-mono text-[11px] text-foreground">
                {secret.secret}
              </div>
              <a href={secret.otpauth_uri} className="text-[11px] text-primary underline-offset-2 hover:underline">
                Открыть ссылку otpauth://
              </a>
              <div className="flex flex-wrap items-center gap-2">
                <div className="w-32">
                  <input
                    value={code}
                    onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                    placeholder="123456"
                    inputMode="numeric"
                    maxLength={6}
                    aria-label="Код из приложения-аутентификатора"
                    className={inputClass + " text-center tabular-nums"}
                  />
                </div>
                <Button size="sm" onClick={confirmSetup} disabled={busy || code.length < 6}>
                  Подтвердить
                </Button>
                <Button size="sm" variant="ghost" onClick={() => { setSecret(null); setCode(""); }} disabled={busy}>
                  Отмена
                </Button>
              </div>
            </div>
          )}

          {recoveryCodes.length > 0 && (
            <div className="rounded-lg border border-warning/30 bg-warning-soft p-3 text-xs">
              <div className="font-medium text-warning">Резервные коды - сохраните, показываются один раз</div>
              <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] text-foreground">
                {recoveryCodes.map((c) => <span key={c}>{c}</span>)}
              </div>
            </div>
          )}

          {me?.enrolled && !disableMode && (
            <Button variant="secondary" size="sm" onClick={() => setDisableMode(true)} disabled={busy}>
              Отключить у себя
            </Button>
          )}
          {disableMode && (
            <div className="flex flex-wrap items-center gap-2">
              <div className="w-48">
                <input
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="Код или резервный код"
                  aria-label="Код или резервный код"
                  className={inputClass}
                />
              </div>
              <Button size="sm" variant="danger" onClick={disableMine} disabled={busy || !code}>
                Отключить
              </Button>
              <Button size="sm" variant="ghost" onClick={() => { setDisableMode(false); setCode(""); }} disabled={busy}>
                Отмена
              </Button>
            </div>
          )}
        </div>

        {message && <span className="text-sm text-muted-foreground">{message}</span>}
      </CardBody>
    </Card>
  );
}
