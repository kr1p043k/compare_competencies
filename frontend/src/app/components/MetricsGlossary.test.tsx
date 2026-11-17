/**
 * MetricsGlossary: сворачиваемая панель расшифровки метрик.
 * aria-expanded/aria-controls, развёртывание и сворачивание, compact-проп.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { METRICS, MetricsGlossary } from "./MetricsGlossary";

const BUTTON_NAME = "Как читать метрики";
const PANEL_ID = "metrics-glossary";

describe("MetricsGlossary", () => {
  it("стартует свёрнутым: aria-expanded=false, панели нет", () => {
    const { container } = render(<MetricsGlossary />);
    const btn = screen.getByRole("button", { name: BUTTON_NAME });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(btn).toHaveAttribute("aria-controls", PANEL_ID);
    expect(container.querySelector(`#${PANEL_ID}`)).toBeNull();
    // Метрики не проступают, пока панель закрыта.
    expect(screen.queryByText(METRICS[0].label)).not.toBeInTheDocument();
  });

  it("разворачивается по клику: aria-expanded=true, все метрики на месте", async () => {
    const user = userEvent.setup();
    const { container } = render(<MetricsGlossary />);
    const btn = screen.getByRole("button", { name: BUTTON_NAME });
    await user.click(btn);

    expect(btn).toHaveAttribute("aria-expanded", "true");
    const panel = container.querySelector(`#${PANEL_ID}`);
    expect(panel).not.toBeNull();
    for (const m of METRICS) {
      expect(screen.getByText(m.label)).toBeInTheDocument();
      expect(screen.getByText(m.short)).toBeInTheDocument();
    }
    // Итоговая пометка про дисциплины без навыков.
    expect(screen.getByText(/отсутствие данных, а не результат/)).toBeInTheDocument();
  });

  it("сворачивается вторым кликом", async () => {
    const user = userEvent.setup();
    const { container } = render(<MetricsGlossary />);
    const btn = screen.getByRole("button", { name: BUTTON_NAME });
    await user.click(btn);
    await user.click(btn);

    expect(btn).toHaveAttribute("aria-expanded", "false");
    expect(container.querySelector(`#${PANEL_ID}`)).toBeNull();
  });

  it("compact меняет размер текста: text-xs вместо text-sm", () => {
    const compact = render(<MetricsGlossary compact />);
    const full = render(<MetricsGlossary />);
    expect(compact.container.firstElementChild?.className).toContain("text-xs");
    expect(full.container.firstElementChild?.className).toContain("text-sm");
  });
});

describe("METRICS: справочник метрик", () => {
  it("содержит 9 метрик с заполненными полями и уникальными ключами", () => {
    expect(METRICS).toHaveLength(9);
    const keys = METRICS.map((m) => m.key);
    expect(new Set(keys).size).toBe(keys.length);
    for (const m of METRICS) {
      expect(m.label.trim()).not.toBe("");
      expect(m.short.trim()).not.toBe("");
      expect(m.detail.trim()).not.toBe("");
    }
  });

  it("ключи покрывают метрики отчёта, используемые на экране", () => {
    expect(METRICS.map((m) => m.key)).toEqual(
      expect.arrayContaining([
        "market_top_coverage",
        "coverage_ratio",
        "quality_coverage",
        "strong_coverage",
        "gaps",
      ]),
    );
  });
});