/**
 * AnalysisTab (7.5.4): заголовок, бейджи запроса/городов, список профессий,
 * загрузка трендов профессии, ошибка и пустое состояние.
 * useApi и motion замоканы; сеть трендов - через lib/api.request.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { useApi } from "../../lib/useApi";
import { AnalysisTab } from "./AnalysisTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/useApi", () => ({ useApi: vi.fn() }));
vi.mock("motion/react", () => ({
  motion: { div: ({ children }: { children?: ReactNode }) => <div>{children}</div> },
}));
vi.mock("./ui/select", () => ({
  Select: ({
    value,
    onValueChange,
    children,
  }: {
    value: string;
    onValueChange?: (v: string) => void;
    children?: ReactNode;
  }) => (
    <select
      aria-label="profession"
      value={value}
      onChange={(e) => onValueChange?.(e.target.value)}
    >
      {children}
    </select>
  ),
  SelectContent: ({ children }: { children?: ReactNode }) => <>{children}</>,
  SelectItem: ({ value, children }: { value: string; children?: ReactNode }) => (
    <option value={value}>{children}</option>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null,
}));

const requestMock = vi.mocked(request);
const useApiMock = vi.mocked(useApi);

beforeEach(() => {
  useApiMock.mockReturnValue({
    data: { professions: [{ name: "Backend-разработчик" }, { name: "Аналитик" }] },
    loading: false,
    error: null,
  } as never);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const text = () => document.body.textContent ?? "";

describe("AnalysisTab", () => {
  it("заголовок и список профессий из useApi", () => {
    render(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    expect(screen.getByText("Анализ компетенций")).toBeTruthy();
    expect(screen.getByText("Backend-разработчик")).toBeTruthy();
    expect(screen.getByText("Аналитик")).toBeTruthy();
    expect(useApiMock).toHaveBeenCalledWith("/api/trends/professions");
  });

  it("бейджи запроса и городов по pipeline-параметрам", () => {
    const { rerender } = render(
      <AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} pipelineQuery="python" pipelineRegions="1,2,3" />,
    );
    expect(text()).toContain("Запрос: python");
    expect(text()).toContain("Города: 3");

    rerender(
      <AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} pipelineRegions="0" />,
    );
    expect(text()).toContain("Весь рынок");
    expect(text()).not.toContain("Города:");

    rerender(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    expect(text()).not.toContain("Весь рынок");
    expect(text()).not.toContain("Запрос:");
  });

  it("выбор профессии грузит тренды и рисует навыки", async () => {
    requestMock.mockResolvedValue({
      skills: [
        { skill: "Python", frequency: 120 },
        { skill: "SQL", frequency: 90 },
      ],
    } as never);
    render(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("profession"), { target: { value: "Аналитик" } });
    await waitFor(() => expect(screen.getByText("Python")).toBeTruthy());
    expect(requestMock).toHaveBeenCalledWith(
      "/api/trends/by-profession?profession=%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D1%82%D0%B8%D0%BA&limit=30",
    );
    expect(text()).toContain("×120");
  });

  it("ошибка загрузки трендов -> role=alert", async () => {
    requestMock.mockRejectedValue(new Error("Тренды недоступны"));
    render(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("profession"), { target: { value: "Аналитик" } });
    const alert = await waitFor(() => screen.getByRole("alert"));
    expect(alert.textContent).toContain("Тренды недоступны");
  });

  it("пустой ответ -> пустое состояние для выбранной профессии", async () => {
    requestMock.mockResolvedValue({ skills: [] } as never);
    render(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("profession"), { target: { value: "Аналитик" } });
    await waitFor(() =>
      expect(screen.getByText("Нет данных для этой профессии")).toBeTruthy(),
    );
  });

  it("без выбора профессии пустого состояния нет", () => {
    render(<AnalysisTab selectedProfile="base" onProfileChange={vi.fn()} />);
    expect(screen.queryByText("Нет данных для этой профессии")).toBeNull();
  });
});