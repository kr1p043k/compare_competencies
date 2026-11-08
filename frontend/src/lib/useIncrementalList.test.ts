import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useIncrementalList } from "./useIncrementalList";

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];

  readonly observed: HTMLElement[] = [];

  observe = vi.fn((node: HTMLElement) => {
    this.observed.push(node);
  });
  unobserve = vi.fn();
  disconnect = vi.fn();

  constructor(private readonly callback: IntersectionObserverCallback) {
    MockIntersectionObserver.instances.push(this);
  }

  trigger(isIntersecting: boolean) {
    const entries = [{ isIntersecting }] as IntersectionObserverEntry[];
    this.callback(entries, this as unknown as IntersectionObserver);
  }
}

interface Props<T> {
  items: T[];
  pageSize: number;
}

beforeEach(() => {
  MockIntersectionObserver.instances = [];
  vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const seq = (n: number) => Array.from({ length: n }, (_, i) => i + 1);

const attachSentinel = (ref: (node: HTMLElement | null) => void) => {
  const node = document.createElement("div");
  act(() => ref(node));
  return node;
};

const lastObserver = () => MockIntersectionObserver.instances.at(-1);

describe("useIncrementalList: окно показа", () => {
  it("показывает первую страницу размером pageSize", () => {
    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(100), pageSize: 24 },
    });

    expect(result.current.visible).toBe(24);
    expect(result.current.hasMore).toBe(true);
  });

  it("обрезает начальное окно по длине массива", () => {
    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(10), pageSize: 24 },
    });

    expect(result.current.visible).toBe(10);
    expect(result.current.hasMore).toBe(false);
  });

  it("loadMore добавляет по pageSize за вызов", () => {
    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(100), pageSize: 24 },
    });

    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(48);

    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(72);
    expect(result.current.hasMore).toBe(true);
  });

  it("loadMore останавливается в конце и не даёт hasMore", () => {
    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(30), pageSize: 24 },
    });

    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(30);
    expect(result.current.hasMore).toBe(false);

    // Повторный вызов в конце списка - no-op.
    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(30);
    expect(result.current.hasMore).toBe(false);
  });

  it("некорректный pageSize трактуется как 1", () => {
    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(5), pageSize: 0 },
    });

    expect(result.current.visible).toBe(1);
    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(2);
  });
});

describe("useIncrementalList: сброс при смене items", () => {
  const render = (items: number[]) =>
    renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items, pageSize: 24 },
    });

  it("сбрасывает окно до pageSize на новом массиве", () => {
    const { result, rerender } = render(seq(100));

    act(() => result.current.loadMore());
    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(72);

    const loadMoreBefore = result.current.loadMore;

    // Смена фильтров/поиска: сервер прислал новый массив.
    rerender({ items: seq(80), pageSize: 24 });
    expect(result.current.visible).toBe(24);
    expect(result.current.hasMore).toBe(true);

    // loadMore стабилен: смена items не должна переподписывать observer.
    expect(result.current.loadMore).toBe(loadMoreBefore);

    // И окно не может выйти за короткий массив.
    rerender({ items: seq(18), pageSize: 24 });
    expect(result.current.visible).toBe(18);
    expect(result.current.hasMore).toBe(false);
  });

  it("не сбрасывает окно при том же массиве", () => {
    const items = seq(100);
    const { result, rerender } = render(items);

    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(48);

    rerender({ items, pageSize: 24 });
    expect(result.current.visible).toBe(48);
  });
});

describe("useIncrementalList: sentinel и IntersectionObserver", () => {
  const render = () =>
    renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(100), pageSize: 24 },
    });

  it("подключает observer к sentinel-узлу", () => {
    const { result } = render();

    const node = attachSentinel(result.current.sentinelRef);

    const observer = lastObserver();
    expect(observer).toBeDefined();
    expect(observer?.observed).toEqual([node]);
  });

  it("пересечение sentinel догружает порцию, пустое - нет", () => {
    const { result } = render();
    attachSentinel(result.current.sentinelRef);

    act(() => lastObserver()?.trigger(false));
    expect(result.current.visible).toBe(24);

    act(() => lastObserver()?.trigger(true));
    expect(result.current.visible).toBe(48);
  });

  it("отключается вместе с узлом и подключается к новому", () => {
    const { result } = render();

    const first = attachSentinel(result.current.sentinelRef);
    act(() => result.current.sentinelRef(null));
    expect(lastObserver()?.disconnect).toHaveBeenCalled();
    expect(MockIntersectionObserver.instances).toHaveLength(1);

    const second = attachSentinel(result.current.sentinelRef);
    const observer = lastObserver();
    expect(observer).not.toBe(MockIntersectionObserver.instances[0]);
    expect(observer?.observed).toEqual([second]);
    expect(first).not.toBe(second);
  });

  it("не плодит observer на ререндерах с тем же массивом", () => {
    const items = seq(100);
    const { result, rerender } = render();

    const sentinelBefore = result.current.sentinelRef;
    attachSentinel(result.current.sentinelRef);
    rerender({ items, pageSize: 24 });
    rerender({ items, pageSize: 24 });

    // Стабильный callback-ref: React не будет перевешивать observer заново.
    expect(result.current.sentinelRef).toBe(sentinelBefore);
    expect(MockIntersectionObserver.instances).toHaveLength(1);
    expect(MockIntersectionObserver.instances[0].disconnect).not.toHaveBeenCalled();
  });

  it("работает и без IntersectionObserver в окружении", () => {
    vi.stubGlobal("IntersectionObserver", undefined);

    const { result } = renderHook((p: Props<number>) => useIncrementalList(p.items, p.pageSize), {
      initialProps: { items: seq(50), pageSize: 24 },
    });

    expect(() => attachSentinel(result.current.sentinelRef)).not.toThrow();
    expect(MockIntersectionObserver.instances).toHaveLength(0);
    expect(result.current.visible).toBe(24);

    // Кнопка остаётся рабочим путём догрузки.
    act(() => result.current.loadMore());
    expect(result.current.visible).toBe(48);
  });
});
