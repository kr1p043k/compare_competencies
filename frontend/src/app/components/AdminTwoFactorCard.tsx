import { useCallback, useEffect, useState } from "react";

import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import { Badge, Card, CardBody, CardHeader, ErrorState, Skeleton } from "../../design";

/** Состояние одного feature-флага, как его отдаёт /api/admin/features. */
interface FeatureFlagState {
  enabled: boolean;
  default: boolean;
  source: "env" | "file";
}

interface FeatureFlagsResponse {
  features: { totp_2fa_enabled?: FeatureFlagState };
}

const TOTP_FLAG = "totp_2fa_enabled";

/**
 * Карточка админ-консоли: мастер-переключатель двухфакторной аутентификации.
 *
 * Флаг живёт на сервере (env переопределяется файлом features.json):
 * - GET  /api/admin/features                - состояние всех флагов;
 * - PUT  /api/admin/features/totp_2fa_enabled - переключить и получить новое состояние.
 */
export function AdminTwoFactorCard() {
  const [flag, setFlag] = useState<FeatureFlagState | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setFailed(false);
    setMsg("");
    try {
      const d = await request<FeatureFlagsResponse>("/api/admin/features");
      const next = d.features?.totp_2fa_enabled ?? null;
      setFlag(next);
      if (!next) setFailed(true);
    } catch (e) {
      reportError("AdminTwoFactorCard:load", e);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toggle = async () => {
    if (!flag || busy) return;
    setBusy(true);
    setMsg("");
    try {
      const d = await request<FeatureFlagsResponse>(`/api/admin/features/${TOTP_FLAG}`, {
        method: "PUT",
        body: { enabled: !flag.enabled },
      });
      const next = d.features?.totp_2fa_enabled;
      if (next) setFlag(next);
      setMsg("Сохранено");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Ошибка сохранения");
    } finally {
      setBusy(false);
    }
  };

  const sourceNote = flag?.source === "file" ? "изменено вручную" : "значение из окружения";

  return (
    <Card id="admin-2fa">
      <CardHeader
        title="Двухфакторная аутентификация"
        description="Мастер-переключатель второго фактора (TOTP) для всех пользователей"
      />
      <CardBody className="space-y-3">
        {loading && (
          <div className="space-y-2" aria-label="Загрузка состояния 2FA">
            <Skeleton className="h-6 w-64" />
            <Skeleton className="h-4 w-72" />
          </div>
        )}
        {failed && !loading && (
          <ErrorState
            title="Не удалось загрузить состояние 2FA"
            description="Проверьте подключение к серверу и повторите"
            onRetry={() => void load()}
          />
        )}
        {flag && !loading && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={flag.enabled ? "success" : "neutral"}>
                  {flag.enabled ? "включена" : "выключена"}
                </Badge>
                <span className="text-xs text-muted-foreground">{sourceNote}</span>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={flag.enabled}
                aria-label="Мастер-переключатель двухфакторной аутентификации"
                onClick={() => void toggle()}
                disabled={busy}
                className={`inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 ${
                  flag.enabled ? "bg-primary" : "bg-muted"
                }`}
              >
                <span
                  aria-hidden="true"
                  className={`ml-0.5 inline-block size-5 rounded-full bg-white shadow transition-transform ${
                    flag.enabled ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
            </div>
            <p className="text-sm">
              {flag.enabled ? (
                <>Второй фактор включён: при входе запрашивается код из приложения-аутентификатора.</>
              ) : (
                <span className="text-muted-foreground">
                  Второй фактор сейчас выключен: вход по паролю. Настроенные ранее ключи сохранены.
                </span>
              )}
            </p>
            {(busy || msg) && (
              <p className="text-xs text-muted-foreground" role="status">
                {busy ? "Сохраняю…" : msg}
              </p>
            )}
          </>
        )}
      </CardBody>
    </Card>
  );
}
