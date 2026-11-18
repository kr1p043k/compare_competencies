import { describe, expect, it } from "vitest";

import { chartColor, chartSeries, confidenceTone, trendColor } from "./tokens";

describe("chart tokens", () => {
  it("keeps the ordered palette aligned with the colour map", () => {
    expect(chartSeries).toEqual([
      chartColor.primary,
      chartColor.sky,
      chartColor.emerald,
      chartColor.amber,
      chartColor.violet,
    ]);
    expect(chartColor.primary).toBe("var(--chart-1)");
    expect(chartColor.violet).toBe("var(--chart-5)");
  });
});

describe("trendColor", () => {
  it.each([
    [1, "var(--success)"],
    [0.01, "var(--success)"],
    [0, "var(--muted-foreground)"],
    [-0.01, "var(--destructive)"],
    [-5, "var(--destructive)"],
  ])("maps %s to %s", (value, expected) => {
    expect(trendColor(value)).toBe(expected);
  });
});

describe("confidenceTone", () => {
  it.each([
    [100, "success"],
    [70, "success"],
    [69.9, "warning"],
    [40, "warning"],
    [39.9, "danger"],
    [0, "danger"],
  ] as const)("maps %s to %s", (percent, expected) => {
    expect(confidenceTone(percent)).toBe(expected);
  });
});
