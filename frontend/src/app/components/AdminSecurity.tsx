import { useEffect, useState } from "react";
import { AlertTriangle, KeyRound, ShieldCheck, ShieldOff } from "lucide-react";

import { ApiError, request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";

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
 * флага недоступно для чтения - тогда показанное состояние это значение по
 * умолчанию (выключено), а не прочитанное из БД.
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
      <CardHeader>
        <CardTitle className="text-lg">
          <ShieldCheck className="size-4 inline mr-2" />
          Двухфакторная аутентификация
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          {state?.enabled ? (
            <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">
              Включена
            </Badge>
          ) : (
            <Badge variant="secondary">Выключена</Badge>
          )}
          <Button
            onClick={toggle}
            disabled={busy || !state}
            variant={state?.enabled ? "destructive" : "default"}
          >
            {state?.enabled ? (
              <><ShieldOff className="size-4 mr-2" />Выключить</>
            ) : (
              <><ShieldCheck className="size-4 mr-2" />Включить</>
            )}
          </Button>
          <span className="text-xs text-muted-foreground">
            подключено админов: {state?.enrolled_admins ?? 0}
          </span>
        </div>

{state?.read_ok === false && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-lg border border-amber-300/60 bg-amber-50/60 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
          >
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <span>
              состояние не читается: действует значение по умолчанию (выключено)
            </span>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Выключено - вход по паролю для всех. Включено - код спрашивается у админов,
          которые подключили TOTP; остальные при входе проходят настройку.
          Выключение не удаляет подключения.
        </p>

        <div className="border-t border-border pt-4 space-y-3">
          <div className="flex items-center gap-2 text-sm">
            <KeyRound className="size-4" />
            Ваша учётная запись:{" "}
            {me?.enrolled ? (
              <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">защищена</Badge>
            ) : (
              <Badge variant="secondary">без кода</Badge>
            )}
          </div>

          {!me?.enrolled && !secret && (
            <Button variant="outline" onClick={startSetup} disabled={busy || !state?.enabled}>
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
              <div className="break-all rounded bg-surface px-2 py-1 font-mono text-[11px]">
                {secret.secret}
              </div>
              <a href={secret.otpauth_uri} className="text-[11px] text-primary underline-offset-2 hover:underline">
                Открыть ссылку otpauth://
              </a>
              <div className="flex flex-wrap items-center gap-2">
                <Input
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                  placeholder="123456"
                  inputMode="numeric"
                  maxLength={6}
                  className="h-9 w-32"
                />
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
            <div className="rounded-lg border border-amber-300/60 bg-amber-50/60 p-3 text-xs dark:border-amber-900 dark:bg-amber-950/40">
              <div className="font-medium">Резервные коды - сохраните, показываются один раз</div>
              <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px]">
                {recoveryCodes.map((c) => <span key={c}>{c}</span>)}
              </div>
            </div>
          )}

          {me?.enrolled && !disableMode && (
            <Button variant="outline" size="sm" onClick={() => setDisableMode(true)} disabled={busy}>
              Отключить у себя
            </Button>
          )}
          {disableMode && (
            <div className="flex flex-wrap items-center gap-2">
              <Input
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Код или резервный код"
                className="h-9 w-48"
              />
              <Button size="sm" variant="destructive" onClick={disableMine} disabled={busy || !code}>
                Отключить
              </Button>
              <Button size="sm" variant="ghost" onClick={() => { setDisableMode(false); setCode(""); }} disabled={busy}>
                Отмена
              </Button>
            </div>
          )}
        </div>

        {message && <span className="text-sm text-muted-foreground">{message}</span>}
      </CardContent>
    </Card>
  );
}
