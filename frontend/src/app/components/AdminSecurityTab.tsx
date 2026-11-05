/**
 * Вкладка «Безопасность» админ-консоли: глобальный переключатель
 * двухфакторной аутентификации администраторов (TOTP).
 *
 * Читает GET /api/admin/security/settings, переключение - PUT с телом
 * {two_factor_enabled}. Включение уходит на сервер сразу, выключение
 * сначала требует подтверждения в диалоге: вход останется по паролю,
 * а уже настроенные секреты сохраняются. Интерфейс оптимистичный:
 * при ошибке PUT состояние откатывается и показывается текст от сервера.
 */
import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ShieldCheck } from "lucide-react";

import { ApiError, request } from "../../lib/api";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  ErrorState,
  Skeleton,
} from "../../design/primitives";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { cn } from "./ui/utils";

/** GET/PUT /api/admin/security/settings. */
type SecuritySettings = {
  two_factor_enabled: boolean;
  enrolled_admins: number;
  active_admins: number;
};

function errMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    if (err.status >= 500) return "Сервер временно недоступен. Попробуйте позже.";
    // 403 «Недостаточно прав» и другие тексты отдаёт сервер.
    return err.message || fallback;
  }
  return "Нет связи с сервером. Проверьте подключение.";
}

/** Мягкая валидация ответа: не доверяем произвольному JSON. */
function parseSettings(raw: unknown): SecuritySettings | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.two_factor_enabled !== "boolean") return null;
  return {
    two_factor_enabled: r.two_factor_enabled,
    enrolled_admins: typeof r.enrolled_admins === "number" ? r.enrolled_admins : 0,
    active_admins: typeof r.active_admins === "number" ? r.active_admins : 0,
  };
}

export function AdminSecurityTab() {
  const [boot, setBoot] = useState<"loading" | "error" | "ready">("loading");
  const [bootError, setBootError] = useState("");
  const [settings, setSettings] = useState<SecuritySettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmOff, setConfirmOff] = useState(false);
  const [notice, setNotice] = useState<{ text: string; ok: boolean } | null>(null);

  const load = useCallback(async () => {
    setBoot("loading");
    setBootError("");
    setNotice(null);
    try {
      const parsed = parseSettings(await request<unknown>("/api/admin/security/settings"));
      if (!parsed) throw new Error("Некорректный ответ сервера");
      setSettings(parsed);
      setBoot("ready");
    } catch (err) {
      setBootError(errMessage(err, "Не удалось загрузить настройки безопасности"));
      setBoot("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  /** PUT переключателя. Оптимистично, с откатом и текстом ошибки от сервера. */
  const apply = async (target: boolean): Promise<boolean> => {
    if (!settings || saving) return false;
    const before = settings;
    setSettings({ ...before, two_factor_enabled: target });
    setSaving(true);
    setNotice(null);
    try {
      const parsed = parseSettings(
        await request<unknown>("/api/admin/security/settings", {
          method: "PUT",
          body: { two_factor_enabled: target },
        }),
      );
      if (!parsed) throw new Error("Некорректный ответ сервера");
      setSettings(parsed);
      setNotice({
        text: target
          ? "Двухфакторная аутентификация включена: при входе администраторам потребуется код."
          : "Двухфакторная аутентификация выключена: вход по паролю, настроенные секреты сохранены.",
        ok: true,
      });
      return true;
    } catch (err) {
      // Откат оптимистичного состояния до прежнего вида счётчиков.
      setSettings(before);
      setNotice({ text: errMessage(err, "Не удалось сохранить настройку"), ok: false });
      return false;
    } finally {
      setSaving(false);
    }
  };

  const onToggle = () => {
    if (!settings || saving) return;
    if (settings.two_factor_enabled) {
      // Выключение необратимо для входа (код больше не спрашивается),
      // поэтому сначала показываем диалог подтверждения.
      setConfirmOff(true);
      return;
    }
    void apply(true);
  };

  const onConfirmOff = async () => {
    // Ошибку PUT показываем в карточке: диалог в любом случае закрываем,
    // состояние переключателя откатывается само.
    await apply(false);
    setConfirmOff(false);
  };

  if (boot === "loading") {
    return (
      <Card>
        <CardHeader
          title="Двухфакторная аутентификация"
          description="Глобальный переключатель для аккаунтов администраторов"
        />
        <CardBody className="space-y-3" aria-label="Загружаю настройки безопасности">
          <Skeleton className="h-6 w-64" />
          <Skeleton className="h-4 w-72" />
          <Skeleton className="h-4 w-56" />
        </CardBody>
      </Card>
    );
  }

  if (boot === "error") {
    return (
      <Card>
        <CardHeader
          title="Двухфакторная аутентификация"
          description="Глобальный переключатель для аккаунтов администраторов"
        />
        <CardBody>
          <ErrorState description={bootError} onRetry={() => void load()} />
        </CardBody>
      </Card>
    );
  }

  if (!settings) return null;

  const on = settings.two_factor_enabled;

  return (
    <Card>
      <CardHeader
        title={
          <span className="flex flex-wrap items-center gap-2">
            <ShieldCheck className="size-5 text-primary" />
            Двухфакторная аутентификация
            <Badge tone={on ? "success" : "warning"}>{on ? "включена" : "выключена"}</Badge>
          </span>
        }
        description="Глобальный переключатель для аккаунтов администраторов"
      />
      <CardBody className="space-y-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            role="switch"
            aria-checked={on}
            aria-busy={saving || undefined}
            aria-labelledby="two-factor-state-label"
            disabled={saving}
            onClick={onToggle}
            className={cn(
              "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full border border-transparent transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              "disabled:cursor-not-allowed disabled:opacity-60",
              on ? "bg-primary" : "bg-muted",
            )}
          >
            <span
              className={cn(
                "pointer-events-none mx-0.5 block size-5 rounded-full bg-background shadow-sm transition-transform",
                on ? "translate-x-5" : "translate-x-0",
              )}
            />
          </button>
          <span id="two-factor-state-label" className="text-sm font-medium text-foreground">
            {on ? "Включена для всех администраторов" : "Выключена"}
          </span>
        </div>

        <p className="text-sm text-muted-foreground">
          Настроили: {settings.enrolled_admins} из {settings.active_admins} активных администраторов
        </p>

        <p className="text-sm text-muted-foreground">
          {on
            ? "Каждый администратор может подключить приложение-аутентификатор в своём профиле. При входе, кроме пароля, потребуется одноразовый код."
            : "Сейчас вход выполняется только по паролю. Уже настроенные секреты сохраняются и снова потребуются после включения защиты."}
        </p>

        <details className="group rounded-lg border border-border bg-muted/40 px-4 py-3">
          <summary className="flex cursor-pointer select-none items-center gap-1.5 rounded-sm text-sm font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <ChevronDown className="size-4 text-muted-foreground transition-transform group-open:rotate-180" />
            Подробнее
          </summary>
          <div className="mt-3 space-y-2 text-xs text-muted-foreground">
            <p>
              <span className="font-medium text-foreground">При включении:</span> при входе
              администраторам потребуется одноразовый код из приложения-аутентификатора; подключить
              его можно в профиле. Уже настроенные секреты сохраняются.
            </p>
            <p>
              <span className="font-medium text-foreground">При выключении:</span> вход останется
              по паролю, настроенные секреты не удаляются и снова вступят в силу после включения
              защиты.
            </p>
          </div>
        </details>

        {notice && (
          <p
            role="status"
            aria-live="polite"
            className={notice.ok ? "text-sm text-success" : "text-sm text-destructive"}
          >
            {notice.text}
          </p>
        )}
      </CardBody>

      <Dialog
        open={confirmOff}
        onOpenChange={(open) => {
          if (!saving) setConfirmOff(open);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Выключить двухфакторную аутентификацию?</DialogTitle>
            <DialogDescription>
              Вход останется по паролю: одноразовый код при входе больше не потребуется. Уже
              настроенные секреты сохранятся и снова вступят в силу после включения защиты.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setConfirmOff(false)} disabled={saving}>
              Отмена
            </Button>
            <Button
              variant="danger"
              onClick={() => void onConfirmOff()}
              disabled={saving}
              aria-busy={saving || undefined}
            >
              Выключить
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
