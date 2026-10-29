/**
 * Единая обработка ошибок фронтенда.
 *
 * Раньше в 49 местах стояло `.catch(() => {})` или `catch {}` — сбой сети,
 * 500-й ответ или битый JSON исчезали бесследно, и пользователь видел
 * пустой экран без объяснения. Здесь ошибка всегда доходит до консоли
 * (и до /api/log), а вызывающий код решает, показывать ли её в UI.
 */
import { ApiError } from "./api";

const listeners = new Set<(message: string) => void>();

/** Подписка на ошибки для показа в UI (тосты/баннеры). */
export function onError(listener: (message: string) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notify(message: string): void {
  for (const listener of listeners) {
    try {
      listener(message);
    } catch {
      /* обработчик UI сам упал — не роняем остальные */
    }
  }
}

/**
 * Логирует ошибку с контекстом и уведомляет подписчиков.
 * Возвращает текст, пригодный для показа пользователю.
 */
export function reportError(context: string, error: unknown, userVisible = false): string {
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : String(error);

  // eslint-disable-next-line no-console
  console.error(`[${context}]`, error);

  void logToBackend(context, message);

  if (userVisible) notify(message);
  return message;
}

async function logToBackend(context: string, message: string): Promise<void> {
  try {
    await fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        level: "error",
        message: context,
        data: { detail: message },
        timestamp: new Date().toISOString(),
      }),
      keepalive: true,
    });
  } catch {
    /* логирование не должно ломать сценарий */
  }
}
