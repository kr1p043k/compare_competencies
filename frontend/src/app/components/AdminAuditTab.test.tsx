/**
 * AdminAuditTab: список событий, фильтр по действию, пустой журнал, экспорт CSV.
 * Сеть мокается на уровне lib/api (request + saveBlob), fetch не вызывается.
 */
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request, saveBlob } from "../../lib/api";
import { AdminAuditTab } from "./AdminAuditTab";

vi.mock("../../lib/api", () => ({ request: vi.fn(), saveBlob: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const requestMock = vi.mocked(request);
const saveBlobMock = vi.mocked(saveBlob);

interface Page {
  entries: unknown[];
  total: number;
  limit: number;
  offset: number;
}

const EMPTY: Page = { entries: [], total: 0, limit: 50, offset: 0 };
let page: Page;

const ENTRIES = [
  { id: "a-1", actor_email: "root@sfedu.ru", actor_role: "admin", action: "user.patch",
    target_type: "user", target_id: "u-teacher", target: "Иван Петров", meta: null,
    ip: "10.0.0.9", request_id: "req-17", created_at: 1735689600 },
  { id: "a-2", actor_email: "op@sfedu.ru", actor_role: "rop", action: "login",
    target_type: "session", target_id: null, target: "", meta: null,
    ip: "10.0.0.10", request_id: null, created_at: 1735693200 },
];

function route(url: string): Promise<unknown> {
  if (url === "/api/admin/audit/actions") {
    return Promise.resolve({ actions: ["user.patch", "login"], total: 2 });
  }
  if (url.startsWith("/api/admin/audit?")) return Promise.resolve(page);
  return Promise.reject(new Error(`unexpected endpoint: ${url}`));
}

/** Запросы списка журнала (справочник actions не считаем). */
const auditCalls = () =>
  requestMock.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url.startsWith("/api/admin/audit?"));

/** Текст Blob через FileReader + TextDecoder; ignoreBOM сохраняет \uFEFF в тексте. */
function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve(new TextDecoder("utf-8", { ignoreBOM: true }).decode(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
    reader.readAsArrayBuffer(blob);
  });
}

describe("AdminAuditTab", () => {
  beforeEach(() => {
    page = { entries: ENTRIES, total: ENTRIES.length, limit: 50, offset: 0 };
    requestMock.mockReset();
    saveBlobMock.mockReset();
    requestMock.mockImplementation(route as never);
  });

  it("показывает события журнала: актор, действие, роль и ip", async () => {
    render(<AdminAuditTab />);

    const row = (await screen.findByText("root@sfedu.ru")).closest("tr") as HTMLElement;
    expect(within(row).getByText("user.patch")).toBeInTheDocument();
    expect(within(row).getByText("10.0.0.9")).toBeInTheDocument();
    expect(within(row).getByText("Админ")).toBeInTheDocument();

    const row2 = screen.getByText("op@sfedu.ru").closest("tr") as HTMLElement;
    expect(within(row2).getByText("login")).toBeInTheDocument();
    expect(within(row2).getByText("РОП")).toBeInTheDocument();
  });

  it("фильтр по действию перезапрашивает журнал с action=", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");
    expect(auditCalls()).toHaveLength(1);

    // Фильтр без debounce: последний запрос несёт полный текст действия.
    await user.type(screen.getByLabelText("Действие"), "user.patch");
    await waitFor(() =>
      expect(auditCalls()[auditCalls().length - 1]).toContain("action=user.patch"),
    );
  });

  it("пустой журнал рендерит заглушку", async () => {
    page = EMPTY;
    render(<AdminAuditTab />);

    expect(await screen.findByText("Журнал пуст")).toBeInTheDocument();
    expect(
      screen.getByText("События появятся после первых действий в системе."),
    ).toBeInTheDocument();
  });

  it("экспорт CSV содержит заголовок и строки текущей страницы", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");

    await user.click(screen.getByRole("button", { name: "Экспорт CSV" }));
    await waitFor(() => expect(saveBlobMock).toHaveBeenCalledTimes(1));
    const [blob, filename] = saveBlobMock.mock.calls[0];

    expect(filename).toMatch(/^audit-\d{4}-\d{2}-\d{2}\.csv$/);
    const text = await readBlob(blob);

    // BOM + CRLF: заголовок и две строки журнала
    expect(text.startsWith("\uFEFF")).toBe(true);
    const lines = text.replace(/^\uFEFF/, "").split("\r\n");
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe("время,актор,роль,действие,тип объекта,id объекта,объект,ip,request_id");
    expect(lines[1]).toContain("root@sfedu.ru");
    expect(lines[1]).toContain("user.patch");
    expect(lines[1]).toContain("u-teacher");
    expect(lines[1]).toContain("10.0.0.9");
    expect(lines[1]).toContain("req-17");
  });

  it("быстрый диапазон «24ч» заполняет С/По и шлёт since/until (ISO)", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");
    expect(auditCalls()).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "24ч" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(2));
    expect(screen.getByRole("button", { name: "24ч" })).toHaveAttribute("aria-pressed", "true");

    const params = new URLSearchParams(auditCalls()[1].split("?")[1]);
    expect(params.get("since")).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    expect(params.get("until")).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    // окно - примерно 24 часа от текущего момента (усечение до минут -> допуск 60 с)
    const now = Date.now();
    expect(now - Date.parse(params.get("until") ?? "")).toBeLessThan(60_000);
    expect(Math.abs(now - 24 * 60 * 60 * 1000 - Date.parse(params.get("since") ?? ""))).toBeLessThan(
      60_000,
    );
    // ручные поля показывают подставленное окно
    expect((screen.getByLabelText("С") as HTMLInputElement).value).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect((screen.getByLabelText("По") as HTMLInputElement).value).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it("«повторить фильтр» пересчитывает окно и перезапрашивает журнал", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");
    await user.click(screen.getByRole("button", { name: "24ч" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(2));

    await user.click(screen.getByRole("link", { name: "повторить фильтр" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(3));

    const params = new URLSearchParams(auditCalls()[2].split("?")[1]);
    expect(params.get("since")).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(params.get("until")).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it("«Всё» очищает окно: поля пустые, в запросе нет since/until", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");
    await user.click(screen.getByRole("button", { name: "24ч" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(2));

    await user.click(screen.getByRole("button", { name: "Всё" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(3));
    expect(screen.getByRole("button", { name: "Всё" })).toHaveAttribute("aria-pressed", "true");

    const params = new URLSearchParams(auditCalls()[2].split("?")[1]);
    expect(params.get("since")).toBeNull();
    expect(params.get("until")).toBeNull();
    expect((screen.getByLabelText("С") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("По") as HTMLInputElement).value).toBe("");
  });

  it("ручная правка «С» сбрасывает быстрый диапазон и прячет «повторить фильтр»", async () => {
    const user = userEvent.setup();
    render(<AdminAuditTab />);
    await screen.findByText("root@sfedu.ru");
    await user.click(screen.getByRole("button", { name: "1ч" }));
    await waitFor(() => expect(auditCalls()).toHaveLength(2));
    expect(screen.getByRole("link", { name: "повторить фильтр" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("С"), { target: { value: "2026-01-01T10:00" } });
    await waitFor(() => expect(auditCalls()).toHaveLength(3));

    expect(screen.getByRole("button", { name: "1ч" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("link", { name: "повторить фильтр" })).toBeNull();
    const params = new URLSearchParams(auditCalls()[2].split("?")[1]);
    expect(params.get("since")).toContain("2026-01-01");
  });
});
