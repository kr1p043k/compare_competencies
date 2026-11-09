import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FlaskConical } from "lucide-react";

import { MetricPair, PageHeader } from "./primitives";

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

describe("PageHeader", () => {
  it("рисует заголовок h1, описание и действия", () => {
    render(
      <PageHeader
        title="Научные тренды"
        description="Навыки из публикаций"
        actions={<button type="button">Обновить</button>}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Научные тренды");
    expect(screen.getByText("Навыки из публикаций")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();
  });

  it("показывает иконку в квадрате, когда она передана", () => {
    const { container } = render(<PageHeader icon={FlaskConical} title="Научные тренды" />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("не рисует иконку, когда она не передана", () => {
    const { container } = render(<PageHeader title="Публикации" />);
    expect(container.querySelector("svg")).toBeNull();
  });
});
