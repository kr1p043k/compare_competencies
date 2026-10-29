/**
 * Хуки загрузки данных: убирают ручные `useState(loading/error/data)`
 * и `useEffect` с `fetch` из компонентов.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, request } from "./api";

export interface QueryState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  /** Повторить запрос (например, после ошибки или действия). */
  refetch: () => void;
}

/**
 * GET-запрос с состоянием загрузки.
 *
 * `deps` — зависимости, при изменении которых запрос повторяется.
 * Если `path` равен `null`, запрос не выполняется (условная загрузка).
 */
export function useApi<T>(path: string | null, deps: unknown[] = []): QueryState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(path !== null);
  const [nonce, setNonce] = useState(0);

  // Защита от гонок: применяем результат только последнего запроса.
  const requestId = useRef(0);

  useEffect(() => {
    if (path === null) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }

    const id = ++requestId.current;
    const controller = new AbortController();
    setLoading(true);

    request<T>(path, { signal: controller.signal })
      .then((result) => {
        if (id !== requestId.current) return;
        setData(result);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (id !== requestId.current) return;
        if (controller.signal.aborted) return;
        setError(cause instanceof ApiError ? cause : new ApiError("Не удалось загрузить данные", 0, cause, path));
      })
      .finally(() => {
        if (id === requestId.current) setLoading(false);
      });

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, nonce, ...deps]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);

  return { data, error, loading, refetch };
}

export interface MutationState<TArgs, TResult> {
  run: (args: TArgs) => Promise<TResult>;
  data: TResult | null;
  error: ApiError | null;
  loading: boolean;
  reset: () => void;
}

/**
 * POST/PUT/DELETE с состоянием. Ошибка не «съедается»: она доступна
 * в `error`, вызывающий код сам решает, как её показать.
 */
export function useMutation<TArgs, TResult>(
  fn: (args: TArgs) => Promise<TResult>,
): MutationState<TArgs, TResult> {
  const [data, setData] = useState<TResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(
    async (args: TArgs): Promise<TResult> => {
      setLoading(true);
      try {
        const result = await fn(args);
        if (mounted.current) {
          setData(result);
          setError(null);
        }
        return result;
      } catch (cause) {
        const err =
          cause instanceof ApiError ? cause : new ApiError("Операция не выполнена", 0, cause, "");
        if (mounted.current) setError(err);
        throw err;
      } finally {
        if (mounted.current) setLoading(false);
      }
    },
    [fn],
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { run, data, error, loading, reset };
}

/** Текст ошибки для показа пользователю. */
export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Неизвестная ошибка";
}
