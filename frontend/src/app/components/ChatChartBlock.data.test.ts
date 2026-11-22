import { describe, expect, it } from "vitest";

import { extractChartData } from "./ChatChartBlock.data";
import type { ChatBlock } from "./ChatBlockView";

describe("extractChartData", () => {
  it("returns null when there is no data array", () => {
    expect(extractChartData({ type: "chart" } as ChatBlock)).toBeNull();
    expect(extractChartData({ type: "chart", data: [] } as ChatBlock)).toBeNull();
  });

  it("turns a primitive series into label/value rows", () => {
    const data = extractChartData({ type: "chart", data: [10, "20", 30] } as ChatBlock);
    expect(data).not.toBeNull();
    expect(data!.labelKey).toBe("label");
    expect(data!.seriesKeys).toEqual(["value"]);
    expect(data!.rows).toEqual([
      { label: "1", value: 10 },
      { label: "2", value: 20 },
      { label: "3", value: 30 },
    ]);
    expect(data!.kind).toBe("bar");
  });

  it("drops non numeric primitives", () => {
    const data = extractChartData({ type: "chart", data: [1, "n/a", 3] } as ChatBlock);
    expect(data!.rows).toHaveLength(2);
  });

  it("reads object rows, detects the label key and numeric series", () => {
    const data = extractChartData({
      type: "chart",
      data: [
        { name: "A", total: 5, extra: 2 },
        { name: "B", total: 7, extra: 4 },
      ],
    } as ChatBlock);
    expect(data!.labelKey).toBe("name");
    expect(data!.seriesKeys).toEqual(["total", "extra"]);
    expect(data!.rows).toEqual([
      { name: "A", total: 5, extra: 2 },
      { name: "B", total: 7, extra: 4 },
    ]);
  });

  it("prefers declared series and coerces string numbers", () => {
    const data = extractChartData({
      type: "chart",
      data: [{ label: "A", total: "5", ignored: 99 }],
      series: [{ key: "total" }, "ignored"],
    } as ChatBlock);
    expect(data!.seriesKeys).toEqual(["total", "ignored"]);
    expect(data!.rows[0]).toEqual({ label: "A", total: 5, ignored: 99 });
  });

  it("uses the label of a numeric-only key as string and fills gaps with 0", () => {
    const data = extractChartData({
      type: "chart",
      data: [{ x: "A", v: 1 }, { x: "B", v: null }],
      series: ["v"],
    } as ChatBlock);
    expect(data!.labelKey).toBe("x");
    expect(data!.rows[1]).toEqual({ x: "B", v: 0 });
  });

  it("falls back to the first non numeric string key as the label", () => {
    const data = extractChartData({
      type: "chart",
      data: [{ city: "Moscow", score: 1 }],
      series: ["score"],
    } as ChatBlock);
    expect(data!.labelKey).toBe("city");
  });

  it("returns null when object rows have no numeric series", () => {
    expect(
      extractChartData({ type: "chart", data: [{ name: "A", note: "x" }] } as ChatBlock),
    ).toBeNull();
  });

  it("honours the chart type, defaulting to bar", () => {
    const rows = [{ name: "A", v: 1 }];
    expect(extractChartData({ type: "chart", data: rows, chart_type: "line" } as ChatBlock)!.kind).toBe("line");
    expect(extractChartData({ type: "chart", data: rows, chartType: "AREA" } as ChatBlock)!.kind).toBe("area");
    expect(extractChartData({ type: "chart", data: rows, kind: "pie" } as ChatBlock)!.kind).toBe("pie");
    expect(extractChartData({ type: "chart", data: rows, variant: "weird" } as ChatBlock)!.kind).toBe("bar");
  });

  it("takes the title from title, name or caption", () => {
    const rows = [{ name: "A", v: 1 }];
    expect(extractChartData({ type: "chart", data: rows, title: "T" } as ChatBlock)!.title).toBe("T");
    expect(extractChartData({ type: "chart", data: rows, name: "N" } as ChatBlock)!.title).toBe("N");
    expect(extractChartData({ type: "chart", data: rows, caption: "C" } as ChatBlock)!.title).toBe("C");
  });

  it("caps rows at 30", () => {
    const data = Array.from({ length: 40 }, (_, i) => ({ name: `r${i}`, v: i }));
    const parsed = extractChartData({ type: "chart", data } as ChatBlock);
    expect(parsed!.rows).toHaveLength(30);
  });
});