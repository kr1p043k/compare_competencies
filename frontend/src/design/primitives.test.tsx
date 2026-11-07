import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MetricPair } from "./primitives";

describe("MetricPair", () => {
  it("рисует основное значение и парную метрику с подписью", () => {
    render(
      <MetricPair
        primaryValue="55%"
        primaryTitle="Покрытие"
        secondaryValue="40%"
        secondaryLabel="кач."
        secondaryTitle="Взвешенное покрытие"
      />,
    );
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("кач.: 40%")).toBeInTheDocument();
    expect(screen.getByTitle("Покрытие")).toBeInTheDocument();
    expect(screen.getByTitle("Взвешенное покрытие")).toBeInTheDocument();
  });

  it("показывает только основное значение, когда пары нет в payload", () => {
    render(<MetricPair primaryValue="55%" />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    // Одиночный процент без «кач.» - допустимо только если поля реально нет.
    expect(screen.queryByText(/кач\./)).not.toBeInTheDocument();
  });

  it("не рисует парную метрику при null", () => {
    render(<MetricPair primaryValue="55%" secondaryValue={null} secondaryLabel="кач." />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.queryByText(/кач\./)).not.toBeInTheDocument();
  });

  it("рисует парное значение без подписи, если label не передан", () => {
    render(<MetricPair primaryValue="55%" secondaryValue="40%" />);
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("40%")).toBeInTheDocument();
    expect(screen.queryByText(/: 40%/)).not.toBeInTheDocument();
  });
});
