/**
 * DirectionInsights (7.5.4): сводные блоки направления.
 * Проверяем условный рендер блоков, приоритеты рекомендаций, приписки
 * происхождения, склонения и динамику навыков.
 */
import { cleanup, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { DirectionInsights } from "./DirectionInsights";

afterEach(cleanup);

type Props = ComponentProps<typeof DirectionInsights>;

function analysis(overrides: Record<string, unknown> = {}): Props["analysis"] {
  return {
    recommendations: [],
    top_emerging_across_all: [],
    top_cross_discipline_gaps: [],
    ...overrides,
  } as unknown as Props["analysis"];
}

const text = () => document.body.textContent ?? "";

describe("DirectionInsights", () => {
  it("без данных не рисует блоки рекомендаций", () => {
    render(<DirectionInsights analysis={analysis()} />);
    expect(text()).not.toContain("Что улучшить");
    // Блоки «не хватает в плане» и разрывов рендерятся всегда (details),
    // тренды - только при наличии данных.
    expect(text()).toContain("Не хватает в плане");
    expect(text()).toContain("Разрывы между планом и рынком");
    expect(text()).not.toContain("Растущие и падающие навыки");
  });

  it("рекомендации рендерятся с приоритетом, типом и текстом", () => {
    render(
      <DirectionInsights
        analysis={analysis({
          recommendations: [
            { priority: "high", type: "gap", message: "Добавить практику по SQL" },
            { priority: "medium", type: "market", message: "Усилить Docker" },
            { priority: "low", type: "plan", message: "Пересмотреть РПД" },
          ],
        })}
      />,
    );
    expect(screen.getByText("Что улучшить")).toBeTruthy();
    expect(screen.getByText("Добавить практику по SQL")).toBeTruthy();
    expect(screen.getByText("Усилить Docker")).toBeTruthy();
    expect(screen.getByText("Пересмотреть РПД")).toBeTruthy();
    expect(screen.getByText("high")).toBeTruthy();
    expect(screen.getByText("medium")).toBeTruthy();
    expect(screen.getByText("low")).toBeTruthy();
  });

  it("навыки рынка: частота, приписка происхождения и динамика", () => {
    const { rerender } = render(
      <DirectionInsights
        analysis={analysis({
          top_emerging_across_all: [{ skill: "Kafka", frequency: 42 }],
        })}
      />,
    );
    expect(screen.getByText("Kafka")).toBeTruthy();
    expect(text()).toContain("42 вк.");
    // Без change_pct приписки о динамике нет.
    expect(text()).not.toContain("динамика спроса по срезам рынка");

    rerender(
      <DirectionInsights
        analysis={analysis({
          top_emerging_across_all: [
            { skill: "Kafka", frequency: 42, change_pct: 12.4, basis: "7d" },
            { skill: "Redis", frequency: 15, change_pct: -3 },
          ],
        })}
      />,
    );
    expect(text()).toContain("динамика спроса по срезам рынка");
    expect(screen.getByText("+12%")).toBeTruthy();
    expect(screen.getByText("-3%")).toBeTruthy();
  });

  it("разрывы: склонение числа дисциплин", () => {
    render(
      <DirectionInsights
        analysis={analysis({
          top_cross_discipline_gaps: [
            { skill: "Fortran", disciplines: 1 },
            { skill: "Pascal", disciplines: 2 },
            { skill: "Delphi", disciplines: 5 },
          ],
        })}
      />,
    );
    expect(text()).toContain("1 дисциплина");
    expect(text()).toContain("2 дисциплины");
    expect(text()).toContain("5 дисциплин");
  });

  it("тренды: растущие и падающие с знаками", () => {
    render(
      <DirectionInsights
        analysis={analysis({
          trends: {
            rising: [{ skill: "Rust", change_pct: 25 }],
            declining: [{ skill: "jQuery", change_pct: -40 }],
          },
        })}
      />,
    );
    expect(screen.getByText("Растущие и падающие навыки")).toBeTruthy();
    expect(screen.getByText("Растущие навыки")).toBeTruthy();
    expect(screen.getByText("Падающие навыки")).toBeTruthy();
    expect(text()).toContain("+25%");
    expect(text()).toContain("-40%");
  });

  it("тренды без массивов не создают секцию со списками", () => {
    render(<DirectionInsights analysis={analysis({ trends: { rising: [], declining: [] } })} />);
    expect(screen.getByText("Растущие и падающие навыки")).toBeTruthy();
    expect(screen.queryByText("Растущие навыки")).toBeNull();
    expect(screen.queryByText("Падающие навыки")).toBeNull();
  });
});