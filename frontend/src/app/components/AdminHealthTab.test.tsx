import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { AdminHealthTab } from "./AdminHealthTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("./AdminLlmCostCard", () => ({
  AdminLlmCostCard: () => <div data-testid="llm-cost-stub" />,
}));

afterEach(cleanup);

const requestMock = vi.mocked(request);

describe("AdminHealthTab", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("renders the LLM cost card and the health card", () => {
    render(<AdminHealthTab />);
    expect(screen.getByTestId("llm-cost-stub")).toBeTruthy();
    expect(screen.getByText("Состояние сервиса")).toBeTruthy();
  });

  it("loads /health and shows status plus cluster versions", async () => {
    requestMock.mockResolvedValue({
      status: "ok",
      version: "1.2.3",
      api_ready: true,
      evaluator: true,
      recommendation_engine: true,
      ltr_model_version: "v7",
      clusters_versions: { base: "b1", dc: null },
    });
    render(<AdminHealthTab />);

    await userEvent.click(screen.getByRole("button", { name: /Проверить состояние/ }));

    expect(requestMock).toHaveBeenCalledWith("/health");
    expect(await screen.findByText("Сервис отвечает")).toBeTruthy();
    expect(screen.getByText("1.2.3")).toBeTruthy();
    expect(screen.getByText("v7")).toBeTruthy();
    expect(screen.getByText("b1")).toBeTruthy();
  });

  it("reports the error message when the call fails", async () => {
    requestMock.mockRejectedValue(new Error("gateway down"));
    render(<AdminHealthTab />);

    await userEvent.click(screen.getByRole("button", { name: /Проверить состояние/ }));

    expect(await screen.findByText("gateway down")).toBeTruthy();
  });
});