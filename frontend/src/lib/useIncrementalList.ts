import { useCallback, useRef, useState } from "react";

/** Насколько заранее sentinel считается «почти в кадре» (предзагрузка до края экрана). */
const SENTINEL_ROOT_MARGIN = "600px 0px";

export interface UseIncrementalListResult {
  /** Сколько элементов из items сейчас можно рендерить. */
  visible: number;
  /** Есть ли в items ещё не показанные элементы. */
  hasMore: boolean;
  /** Показать следующую порцию. Безопасен и когда показывать нечего. */
  loadMore: () => void;
  /** Callback-ref для sentinel-элемента: подключает IntersectionObserver. */
  sentinelRef: (node: HTMLElement | null) => void;
}

interface WindowState<T> {
  source: readonly T[];
  visible: number;
}

/**
 * Инкрементальный показ длинного списка: рендерим `pageSize` элементов,
 * дальше - порциями по `pageSize` через `loadMore()` либо автоматически,
 * когда sentinel доезжает до зоны видимости (IntersectionObserver).
 *
 * При смене `items` (смена фильтров/поиска выдала новый массив) окно
 * сбрасывается до `pageSize`. Некорректный `pageSize` (не число, <= 0)
 * трактуется как 1, чтобы окно не росло бесконтрольно.
 */
export function useIncrementalList<T>(
  items: readonly T[],
  pageSize: number,
): UseIncrementalListResult {
  const safePageSize =
    Number.isFinite(pageSize) && pageSize > 0 ? Math.floor(pageSize) : 1;

  const [state, setState] = useState<WindowState<T>>(() => ({
    source: items,
    visible: safePageSize,
  }));

  // Сброс окна при смене массива. Паттерн «хранение информации из прошлых
  // рендеров» из react.dev: setState прямо в рендере применяется до коммита,
  // поэтому пользователь не видит мигания со старым количеством карточек.
  if (state.source !== items) {
    setState({ source: items, visible: safePageSize });
  }

  const visible = Math.min(state.visible, items.length);
  const hasMore = visible < items.length;

  // loadMore читает длину из state.source через функциональный апдейт,
  // поэтому его идентичность не зависит от смены items: пересоздаваться
  // (и переподписывать observer) не нужно.
  const loadMore = useCallback(() => {
    setState((prev) => {
      const next = Math.min(prev.visible + safePageSize, prev.source.length);
      return next > prev.visible ? { ...prev, visible: next } : prev;
    });
  }, [safePageSize]);

  const observerRef = useRef<IntersectionObserver | null>(null);
  const sentinelRef = useCallback(
    (node: HTMLElement | null) => {
      observerRef.current?.disconnect();
      observerRef.current = null;
      // Без IntersectionObserver (SSR, старые браузеры) остаётся кнопка.
      if (!node || typeof IntersectionObserver === "undefined") return;

      const observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((entry) => entry.isIntersecting)) {
            loadMore();
          }
        },
        { rootMargin: SENTINEL_ROOT_MARGIN },
      );
      observer.observe(node);
      observerRef.current = observer;
    },
    [loadMore],
  );

  return { visible, hasMore, loadMore, sentinelRef };
}
