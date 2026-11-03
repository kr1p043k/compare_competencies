import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Явная очистка после каждого теста. При `singleFork: true` и общем jsdom
// автоочистка RTL иногда не успевает, и разметка предыдущего теста остаётся
// в document: тогда поиск по роли находит несколько элементов и файл падает
// в зависимости от порядка запуска (так флакал VacancyCard).
afterEach(cleanup);

// jsdom не умеет matchMedia, а useTheme вызывает его при монтировании.
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

class ObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords(): [] {
    return [];
  }
}

// Radix и motion/react требуют ResizeObserver / IntersectionObserver.
if (!("ResizeObserver" in globalThis)) {
  (globalThis as unknown as Record<string, unknown>).ResizeObserver = ObserverStub;
}
if (!("IntersectionObserver" in globalThis)) {
  (globalThis as unknown as Record<string, unknown>).IntersectionObserver = ObserverStub;
}