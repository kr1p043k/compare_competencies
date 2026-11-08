import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { barPercent, GapBreakdown } from "./gap-metrics";

describe("GapBreakdown", () => {
  it("рисует только пришедшие поля, подпись уровней без J - честная", () => {
    render(
      <GapBreakdown
        fields={{ gapM: 0.8, gapS: 0.6, demandJ: 0.7, userLevel: 0.1 }}
      />,
    );
    // Подпись строится по реально нарисованным уровням, а не по шаблону.
    expect(screen.getByText("Разрыв (M/S)")).toBeInTheDocument();
    expect(screen.getByText("Спрос (J)")).toBeInTheDocument();
    expect(screen.getByText("Ваш уровень")).toBeInTheDocument();
    expect(screen.queryByText("Важность")).not.toBeInTheDocument();
    expect(screen.queryByText("Кластер")).not.toBeInTheDocument();
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Middle: насколько не хватает до требуемого" }),
    ).toHaveTextContent("80%");
    expect(
      screen.getByRole("meter", { name: "Востребованность на рынке, уровень Junior" }),
    ).toHaveTextContent("70%");
  });

  it("полный набор полей получает подписи из ТЗ и все мини-бары", () => {
    render(
      <GapBreakdown
        fields={{
          gapJ: 1,
          gapM: 0.8,
          gapS: 0.6,
          demandJ: 0.7,
          demandM: 0.75,
          demandS: 0.8,
          userLevel: 0.1,
          importance: 0.8,
          cluster: 0.66,
        }}
      />,
    );
    expect(screen.getByText("Разрыв (J/M/S)")).toBeInTheDocument();
    expect(screen.getByText("Спрос (J/M/S)")).toBeInTheDocument();
    expect(screen.getByText("Важность")).toBeInTheDocument();
    expect(screen.getByText("Кластер")).toBeInTheDocument();
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Junior: насколько не хватает до требуемого" }),
    ).toHaveTextContent("100%");
    expect(
      screen.getByRole("meter", { name: "Релевантность навыка кластеру целевых вакансий" }),
    ).toHaveTextContent("66%");
  });

  it("пустой набор полей ничего не рисует", () => {
    const { container } = render(<GapBreakdown fields={{}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("barPercent: скругление без float-хвостов, ограничение 0..100", () => {
    expect(barPercent(0.66)).toBe(66);
    expect(barPercent(0.8)).toBe(80);
    expect(barPercent(0.1)).toBe(10);
    expect(barPercent(1)).toBe(100);
    expect(barPercent(1.5)).toBe(100);
    expect(barPercent(-0.2)).toBe(0);
  });
});
