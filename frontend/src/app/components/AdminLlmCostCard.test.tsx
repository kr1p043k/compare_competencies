import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { AdminLlmCostCard } from "./AdminLlmCostCard";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

afterEach(cleanup);

const apiMock = vi.mocked(api);

describe("AdminLlmCostCard", () => {
  it("показывает сумму, запросы и таблицу по моделям", async () => {
    apiMock.mockResolvedValue({
      total_usd: 1.2345,
      requests: 42,
      by_model: [{ model: "gpt-oss:120b", requests: 42, cost_usd: 1.2345 }],
      by_day: [],
      reason: null,
    } as never);

    render(<AdminLlmCostCard />);

    expect(await screen.findByText("Всего, $")).toBeInTheDocument();
    expect(screen.getAllByText("1.2345").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("42").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("gpt-oss:120b")).toBeInTheDocument();
    expect(apiMock).toHaveBeenCalledWith("/admin/llm/cost?days=30");
  });

  it("показывает EmptyState, когда нет обращений", async () => {
    apiMock.mockResolvedValue({
      total_usd: 0,
      requests: 0,
      by_model: [],
      by_day: [],
      reason: "нет данных",
    } as never);

    render(<AdminLlmCostCard />);
    expect(await screen.findByText("Нет данных")).toBeInTheDocument();
    expect(screen.getByText("нет данных")).toBeInTheDocument();
  });

  it("показывает ErrorState и повторяет запрос по кнопке", async () => {
    apiMock.mockRejectedValueOnce(new Error("boom"));
    apiMock.mockResolvedValueOnce({
      total_usd: 0,
      requests: 0,
      by_model: [],
      by_day: [],
      reason: null,
    } as never);

    render(<AdminLlmCostCard />);
    expect(await screen.findByText(/Не удалось загрузить/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button"));
    expect(await screen.findByText("Нет данных")).toBeInTheDocument();
    expect(apiMock).toHaveBeenCalledTimes(2);
  });
});
