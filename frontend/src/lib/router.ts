import { useCallback, useEffect, useState } from "react";

/** Все вкладки приложения. Неизвестное значение в адресе откатывается к fallback. */
export const KNOWN_TABS = [
  "vacancies", "data", "visualization", "predictions", "articles",
  "scientific-trends", "publications", "teacher", "students", "monitoring", "logs",
  "admin", "student", "taxonomy", "profile", "help",
] as const;

const PREFIX = "#/";

function readHash(fallback: string): string {
  const raw = window.location.hash.replace(/^#\/?/, "").trim();
  if (!raw) return fallback;
  return (KNOWN_TABS as readonly string[]).includes(raw) ? raw : fallback;
}

/**
 * Текущая вкладка живёт в hash-части адреса: `#/teacher`.
 *
 * Так появились deep links и история (кнопки «назад/вперёд»), которых не было,
 * пока вкладка хранилась только в useState. Отдельная роутер-зависимость не
 * нужна: маршрутов 15, все плоские.
 *
 * `enabled` (из App это isAuth): хук монтируется раньше, чем становится
 * известна авторизация, и раньше `return <LoginPage/>`. Пока enabled=false,
 * адрес не дефолтим и не нормализуем - иначе экран входа подменяет `/` на
 * `#/vacancies` и подсвечивает чужую вкладку. Включили - прежнее поведение.
 */
export function useTabRoute(
  fallback = "vacancies",
  enabled = true,
): [string, (next: string | ((current: string) => string)) => void] {
  const [tab, setTab] = useState(() => readHash(fallback));

  useEffect(() => {
    if (!enabled) return;
    const onHashChange = () => {
      const value = readHash(fallback);
      setTab(value);
      // Мусор в адресе приводим к известной вкладке, иначе ссылку нельзя скопировать.
      if (window.location.hash !== `${PREFIX}${value}`) {
        window.location.replace(`${PREFIX}${value}`);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    if (!window.location.hash) {
      window.location.replace(`${PREFIX}${fallback}`);
    } else {
      onHashChange();
    }
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [fallback, enabled]);

  const navigate = useCallback(
    (next: string | ((current: string) => string)) => {
      const value = typeof next === "function" ? next(readHash(fallback)) : next;
      const target = (KNOWN_TABS as readonly string[]).includes(value) ? value : fallback;
      // Выключенный роутер (экран входа) адрес не пишет: только память.
      if (!enabled || readHash(fallback) === target) {
        setTab(target);
        return;
      }
      window.location.hash = `/${target}`;
    },
    [fallback, enabled],
  );

  return [tab, navigate];
}
