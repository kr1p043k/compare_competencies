/**
 * Одно значение в hash-части адреса: `#/admin?key=value`.
 *
 * Приложение использует hash-роутинг (`#/admin`), поэтому параметры пишутся
 * после таба через `?`: таб из адреса читается router.ts до знака вопроса.
 *
 * Чтение: при монтировании и на `hashchange` (кнопки «назад/вперёд»,
 * внешняя правка адреса). Запись: `history.replaceState` по умолчанию,
 * чтобы ввод и смена фильтров не плодили записи в истории; `pushState`
 * только по явной опции (смена страницы таблицы).
 */
import { useCallback, useEffect, useRef, useState } from "react";

type UrlValue = string | number;

export interface UrlStateSetOptions {
  /** Добавить запись в историю вместо замены текущей (для смены страницы). */
  push?: boolean;
}

interface HashParts {
  path: string;
  query: string;
}

function readHashParts(): HashParts {
  const raw =
    typeof window === "undefined" ? "" : window.location.hash.replace(/^#/, "");
  const sep = raw.indexOf("?");
  return sep === -1
    ? { path: raw || "/", query: "" }
    : { path: raw.slice(0, sep) || "/", query: raw.slice(sep + 1) };
}

function readValue<T extends UrlValue>(raw: string | null, defaultValue: T): T {
  if (raw === null) return defaultValue;
  if (typeof defaultValue === "number") {
    const parsed = Number(raw);
    return raw.trim() !== "" && Number.isFinite(parsed) ? (parsed as T) : defaultValue;
  }
  return raw as T;
}

export function useUrlState<T extends UrlValue>(
  key: string,
  defaultValue: T,
): [T, (next: T | ((current: T) => T), options?: UrlStateSetOptions) => void] {
  const [value, setValue] = useState<T>(() =>
    readValue(new URLSearchParams(readHashParts().query).get(key), defaultValue),
  );
  // Актуальное значение для функциональной формы сеттера без пересоздания колбэка.
  const valueRef = useRef(value);
  valueRef.current = value;

  const setUrlValue = useCallback(
    (next: T | ((current: T) => T), options?: UrlStateSetOptions) => {
      const resolved =
        typeof next === "function" ? (next as (current: T) => T)(valueRef.current) : next;
      const serialized = String(resolved);
      const isDefault = serialized === String(defaultValue);
      const { path, query } = readHashParts();
      const params = new URLSearchParams(query);
      const current = params.get(key);

      if (current !== serialized || (isDefault && current !== null)) {
        if (isDefault) params.delete(key);
        else params.set(key, serialized);

        const nextQuery = params.toString();
        const hash = `#${path}${nextQuery ? `?${nextQuery}` : ""}`;
        if (options?.push) window.history.pushState(null, "", hash);
        else window.history.replaceState(null, "", hash);
      }

      valueRef.current = resolved;
      setValue(resolved);
    },
    [defaultValue, key],
  );

  useEffect(() => {
    const onHashChange = () => {
      const next = readValue(new URLSearchParams(readHashParts().query).get(key), defaultValue);
      valueRef.current = next;
      setValue(next);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [defaultValue, key]);

  return [value, setUrlValue];
}
