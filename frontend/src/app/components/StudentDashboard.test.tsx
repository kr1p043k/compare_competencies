/**
 * StudentDashboard: smoke (7.5.4). Дочерние панели стабятся, история - из мока.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { StudentDashboard } from "./StudentDashboard";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn(() => "ошибка сети") }));
vi.mock("./StudentStartHere", () => ({ StudentStartHere: () => null }));
vi.mock("./StudentKrm", () => ({ StudentKrm: () => null }));
vi.mock("./SelfProfileEditor", () => ({ SelfProfileEditor: () => null }));
vi.mock("./LlmChatPanel", () => ({ LlmChatPanel: () => null }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const requestMock = vi.mocked(request);

describe("StudentDashboard (smoke)", () => {
  it("пустая история -> EmptyState", async () => {
    requestMock.mockResolvedValue({ history: [] } as never);
    render(<StudentDashboard />);
    await waitFor(() => expect(screen.getByText("История пуста")).toBeTruthy());
  });

  it("история рендерится в таблицу", async () => {
    requestMock.mockResolvedValue({
      history: [
        { timestamp: "2026-01-01T10:00:00Z", profession: "Python разработчик",
          region: "0", vacancies_found: 42, profile: "demo" },
      ],
    } as never);
    render(<StudentDashboard />);
    await waitFor(() => expect(screen.getByText("Python разработчик")).toBeTruthy());
    expect(screen.getByText("Весь рынок")).toBeTruthy();
    expect(screen.getByText("Открыть анализ")).toBeTruthy();
  });

  it("ошибка -> ErrorState с повтором", async () => {
    requestMock.mockRejectedValue(new Error("boom"));
    render(<StudentDashboard />);
    await waitFor(() =>
      expect(screen.getByText("Не удалось загрузить историю запросов")).toBeTruthy(),
    );
    expect(screen.getByText("Повторить")).toBeTruthy();
  });
});
