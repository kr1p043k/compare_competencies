import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FlaskConical } from "lucide-react";

import { Button, Meter, MetricPair, PageHeader, Sparkbars } from "./primitives";

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

describe("Meter", () => {
  it("обрезает значение в 0..100 и красит трек/заливку переданными классами", () => {
    const { container } = render(
      <Meter value={150} trackClassName="bg-warning/40" fillClassName="bg-info" />,
    );
    const track = container.firstElementChild as HTMLElement;
    const fill = track.firstElementChild as HTMLElement;
    expect(fill.style.width).toBe("100%");
    expect(track.className).toContain("bg-warning/40");
    expect(fill.className).toContain("bg-info");
  });

  it("держит дефолтные цвета и 0% при undefined/NaN", () => {
    const { container } = render(<Meter value={undefined as unknown as number} />);
    const track = container.firstElementChild as HTMLElement;
    const fill = track.firstElementChild as HTMLElement;
    expect(track.className).toContain("bg-muted");
    expect(fill.className).toContain("bg-primary");
    expect(fill.style.width).toBe("0%");
  });

  it("не уходит в минус при отрицательном значении", () => {
    const { container } = render(<Meter value={-20} />);
    const fill = (container.firstElementChild as HTMLElement).firstElementChild as HTMLElement;
    expect(fill.style.width).toBe("0%");
  });
});

describe("Sparkbars", () => {
  const items = [
    { key: "a", value: 4, label: "03.02", title: "2026-02-03: 4" },
    { key: "b", value: 2, label: "04.02", title: "2026-02-04: 2" },
  ];

  it("считает высоту столбиков относительно максимума", () => {
    const { container } = render(<Sparkbars items={items} />);
    const bars = Array.from(
      container.querySelectorAll<HTMLElement>("div[title^='2026-02']"),
    );
    expect(bars).toHaveLength(2);
    expect(bars[0].style.height).toBe("100%");
    expect(bars[1].style.height).toBe("50%");
  });

  it("рисует значение над столбиком и подпись под ним", () => {
    render(<Sparkbars items={items} />);
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("03.02")).toBeInTheDocument();
    expect(screen.getByText("04.02")).toBeInTheDocument();
  });

  it("не падает на пустом наборе", () => {
    const { container } = render(<Sparkbars items={[]} />);
    expect(container.querySelectorAll("div[title]")).toHaveLength(0);
  });
});

describe("Button", () => {
  it("вариант success красит кнопку в токен success", () => {
    render(<Button variant="success">Создать</Button>);
    const btn = screen.getByRole("button", { name: "Создать" });
    expect(btn.className).toContain("bg-success");
    expect(btn.className).toContain("text-success-foreground");
  });

  it("вариант danger остаётся на токене destructive", () => {
    render(<Button variant="danger">Удалить</Button>);
    const btn = screen.getByRole("button", { name: "Удалить" });
    expect(btn.className).toContain("bg-destructive");
  });
});
