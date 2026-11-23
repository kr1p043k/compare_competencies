/**
 * ChatChartBlock (7.5.4): рендер bar/line/area/pie по ChartData.
 * recharts замокан passthrough со сбором пропсов: проверяем, что выбран
 * правильный тип графика, серии и заголовок; сам recharts не нужен в jsdom.
 */
import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChatChartBlock } from "./ChatChartBlock";
import type { ChartData } from "./ChatChartBlock.data";

const seen = {
  bars: [] as Array<Record<string, unknown>>,
  lines: [] as Array<Record<string, unknown>>,
  areas: [] as Array<Record<string, unknown>>,
  pies: [] as Array<Record<string, unknown>>,
  xTickFormatter: null as null | ((v: unknown) => string),
};

vi.mock("recharts", () => {
  const Pass = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  return {
    ResponsiveContainer: Pass,
    BarChart: Pass,
    LineChart: Pass,
    AreaChart: Pass,
    PieChart: Pass,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    Cell: () => null,
    XAxis: ({ tickFormatter }: { tickFormatter?: (v: unknown) => string }) => {
      seen.xTickFormatter = tickFormatter ?? null;
      return null;
    },
    YAxis: () => null,
    Bar: (props: Record<string, unknown>) => {
      seen.bars.push(props);
      return null;
    },
    Line: (props: Record<string, unknown>) => {
      seen.lines.push(props);
      return null;
    },
    Area: (props: Record<string, unknown>) => {
      seen.areas.push(props);
      return null;
    },
    Pie: (props: Record<string, unknown>) => {
      seen.pies.push(props);
      return null;
    },
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  seen.bars = [];
  seen.lines = [];
  seen.areas = [];
  seen.pies = [];
  seen.xTickFormatter = null;
});

function data(overrides: Partial<ChartData> = {}): ChartData {
  return {
    kind: "bar",
    labelKey: "label",
    seriesKeys: ["a", "b"],
    rows: [
      { label: "x", a: 1, b: 2 },
      { label: "y", a: 3, b: 4 },
    ],
    title: "Навыки",
    ...overrides,
  };
}

describe("ChatChartBlock", () => {
  it("bar: по Bar на каждую серию и заголовок", () => {
    render(<ChatChartBlock data={data()} />);
    expect(screen.getByText("Навыки")).toBeTruthy();
    expect(seen.bars.map((b) => b.dataKey)).toEqual(["a", "b"]);
    expect(seen.lines).toHaveLength(0);
  });

  it("line: по Line на серию", () => {
    render(<ChatChartBlock data={data({ kind: "line" })} />);
    expect(seen.lines.map((l) => l.dataKey)).toEqual(["a", "b"]);
    expect(seen.bars).toHaveLength(0);
  });

  it("area: по Area на серию", () => {
    render(<ChatChartBlock data={data({ kind: "area" })} />);
    expect(seen.areas.map((a) => a.dataKey)).toEqual(["a", "b"]);
  });

  it("pie: один Pie с первой серией и именем по labelKey", () => {
    render(<ChatChartBlock data={data({ kind: "pie", seriesKeys: ["value"] , rows: [{ label: "x", value: 1 }] })} />);
    expect(seen.pies).toHaveLength(1);
    expect(seen.pies[0].dataKey).toBe("value");
    expect(seen.pies[0].nameKey).toBe("label");
  });

  it("без заголовка строка заголовка не рендерится, но график есть", () => {
    render(<ChatChartBlock data={data({ title: "" })} />);
    expect(screen.queryByText("Навыки")).toBeNull();
    expect(seen.bars.length).toBe(2);
  });

  it("тиксы оси усекаются до 12 символов с многоточием", () => {
    render(<ChatChartBlock data={data()} />);
    expect(seen.xTickFormatter).toBeTypeOf("function");
    expect(seen.xTickFormatter!("короткий")).toBe("короткий");
    expect(seen.xTickFormatter!("очень-длинная-категория")).toBe("очень-длинн…");
    expect(seen.xTickFormatter!(123)).toBe("123");
  });
});