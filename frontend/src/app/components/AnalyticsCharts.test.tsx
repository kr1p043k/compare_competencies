import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { AnalyticsCharts } from "./AnalyticsCharts";

vi.mock("../api", () => ({ api: vi.fn() }));

const apiMock = vi.mocked(api);

const TOP = ["Python", "SQL", "Docker"];
const PROFILES: Record<string, string[]> = {
  base: ["python", "sql", "docker"],
  dc: ["python", "sql"],
  top_dc: ["docker"],
};

function route(endpoint: string) {
  if (endpoint.startsWith("/market/top-skills")) {
    return Promise.resolve({ skills: TOP.map((s) => ({ skill: s, weight: 1 })) });
  }
  const m = endpoint.match(/^\/profiles\/(\w+)\?/);
  if (m) return Promise.resolve({ skills: PROFILES[m[1]] ?? [] });
  return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
}

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation(route as never);
});

describe("AnalyticsCharts: блок покрытия топ-50", () => {
  it("свёрнут по умолчанию: компактный свод вместо диаграммы", async () => {
    render(<AnalyticsCharts />);
    expect(await screen.findByText("Покрытие топ-50 навыков рынка по уровням")).toBeInTheDocument();

    const toggle = screen.getByRole("button", { name: "Показать диаграмму" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    expect(
      screen.getByRole("button", { name: "Base (junior): покрытие 100% топ-50, открыть списки навыков" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Data Scientist (middle): покрытие 67% топ-50, открыть списки навыков" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Top (senior): покрытие 33% топ-50, открыть списки навыков" }),
    ).toBeInTheDocument();
    expect(document.getElementById("coverage-chart")).toBeNull();
  });

  it("раскрывает диаграмму по клику и убирает свод", async () => {
    const user = userEvent.setup();
    render(<AnalyticsCharts />);
    await screen.findByText("Покрытие топ-50 навыков рынка по уровням");

    await user.click(screen.getByRole("button", { name: "Показать диаграмму" }));

    const toggle = screen.getByRole("button", { name: "Свернуть" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("coverage-chart")).not.toBeNull();
    expect(screen.queryByRole("button", { name: /открыть списки навыков/ })).not.toBeInTheDocument();

    await user.click(toggle);
    expect(screen.getByRole("button", { name: "Показать диаграмму" })).toHaveAttribute("aria-expanded", "false");
  });

  it("открывает списки навыков кликом по уровню в своде", async () => {
    const user = userEvent.setup();
    render(<AnalyticsCharts />);
    await screen.findByText("Покрытие топ-50 навыков рынка по уровням");

    await user.click(
      screen.getByRole("button", { name: "Base (junior): покрытие 100% топ-50, открыть списки навыков" }),
    );

    expect(await screen.findByText("Покрытие топ-50: 100%")).toBeInTheDocument();
    expect(screen.getByText("Есть (3)")).toBeInTheDocument();
  });

  it("оформляет «профили без навыков» как предупреждение, а не голый текст", async () => {
    apiMock.mockImplementation(((endpoint: string) => {
      if (endpoint.startsWith("/market/top-skills")) {
        return Promise.resolve({ skills: TOP.map((s) => ({ skill: s, weight: 1 })) });
      }
      const m = endpoint.match(/^\/profiles\/(\w+)\?/);
      if (m) return Promise.resolve({ skills: m[1] === "base" ? [] : PROFILES[m[1]] ?? [] });
      return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
    }) as never);

    render(<AnalyticsCharts />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Профили без навыков: base. Перезапустите бэкенд и обновите страницу (Ctrl+F5).",
    );
  });
});
