import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { MarketView } from "./MarketView";

describe("MarketView", () => {
  it("shows the highest-weight skill as the leader", () => {
    render(
      <MarketView
        data={{
          skills: [
            { skill: "python", weight: 0.4 },
            { skill: "sql", weight: 0.9 },
          ],
          total: 2,
          vacancy_count: 1234,
        }}
      />,
    );
    // Топ-навык в StatCard = лидер по весу
    expect(screen.getAllByText("sql").length).toBeGreaterThan(0);
    expect(screen.getByText("1 234")).toBeTruthy();
  });

  it("sorts rows by weight descending", () => {
    const { container } = render(
      <MarketView
        data={{
          skills: [
            { skill: "aaa", weight: 0.1 },
            { skill: "zzz", weight: 0.9 },
          ],
        }}
      />,
    );
    const text = container.textContent || "";
    expect(text.indexOf("zzz")).toBeGreaterThan(-1);
    expect(text.indexOf("zzz")).toBeLessThan(text.indexOf("aaa"));
  });

  it("shows no leader when there are no skills", () => {
    const { container } = render(<MarketView data={{ skills: [] }} />);
    // лидера нет -> стат «Топ-навык» показывает прочерк
    expect(container.textContent).toContain("-");
  });
});
