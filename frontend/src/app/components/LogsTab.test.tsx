/**
 * LogsTab: загрузка логов по кнопке «Обновить», пусто/ошибка/поиск (7.5.4).
 * Живой WebSocket в jsdom недоступен - поэтому проверяем ручную загрузку.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { LogsTab } from "./LogsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
}));

// jsdom не умеет WebSocket: подменяем на «сразу закрытый», чтобы компонент
// ушёл в fallback и не мешал ручной загрузке.
class NoopWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor() {
    // Сразу сообщаем о закрытии: компонент перейдёт в fallback и снимет loading,
    // иначе кнопка «Обновить» останется disabled.
    setTimeout(() => this.onclose?.(), 0);
  }
  close() {}
}

beforeEach(() => {
  // @ts-expect-error test shim
  globalThis.WebSocket = NoopWebSocket;
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const requestMock = vi.mocked(request);
const text = () => document.body.textContent ?? "";

async function loadByButton() {
  const btn = await waitFor(() => {
    const b = screen.getByRole("button", { name: /Обновить/ }) as HTMLButtonElement;
    if (b.disabled) throw new Error("кнопка ещё disabled (loading)");
    return b;
  });
  fireEvent.click(btn);
}

describe("LogsTab", () => {
  it("ручная загрузка рендерит строки логов", async () => {
    requestMock.mockResolvedValue({ lines: ["2026 INFO старт", "2026 ERROR сбой"] } as never);
    render(<LogsTab />);
    expect(screen.getByText("Логи")).toBeTruthy();
    await loadByButton();
    await waitFor(() => expect(text()).toContain("ERROR сбой"));
  });

  it("пустой лог -> «Записей не найдено»", async () => {
    requestMock.mockResolvedValue({ lines: [] } as never);
    render(<LogsTab />);
    await loadByButton();
    await waitFor(() => expect(text()).toContain("Записей не найдено"));
  });

  it("ошибка загрузки показывается в alert", async () => {
    requestMock.mockRejectedValue(new Error("логи недоступны"));
    render(<LogsTab />);
    await loadByButton();
    const alert = await waitFor(() => screen.getByRole("alert"));
    expect(alert.textContent).toMatch(/логи недоступны/);
  });

  it("поиск фильтрует строки", async () => {
    requestMock.mockResolvedValue({
      lines: ["2026 INFO старт", "2026 ERROR сбой", "2026 DEBUG шум"],
    } as never);
    render(<LogsTab />);
    await loadByButton();
    await waitFor(() => expect(text()).toContain("сбой"));
    fireEvent.change(screen.getByPlaceholderText("Поиск по логам..."), {
      target: { value: "старт" },
    });
    await waitFor(() => expect(text()).not.toContain("сбой"));
    expect(text()).toContain("старт");
  });
});