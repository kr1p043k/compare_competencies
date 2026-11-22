/**
 * a11y: AnalyticsCharts без нарушений axe после загрузки покрытия топ-50.
 *
 * Сеть мокается на уровне app/api (обёртка api()), по образцу
 * AnalyticsCharts.test.tsx. Ждём загруженные данные через findBy.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, it, vi } from "vitest";

import { api } from "../api";
import { expectNoA11yViolations } from "../../test/a11y";
import { AnalyticsCharts } from "./AnalyticsCharts";

vi.mock("../api", () => ({ api: vi.fn() }));

const apiMock = vi.mocked(api);

const TOP = ["Python", "SQL", "Docker"];
const PROFILES: Record<string, string[]> = {
  base: ["python", "sql", "docker"],
  dc: ["python", "sql"],
  top_dc: ["docker"],
};

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation(((endpoint: string) => {
    if (endpoint.startsWith("/market/top-skills")) {
      return Promise.resolve({ skills: TOP.map((s) => ({ skill: s, weight: 1 })) });
    }
    const m = endpoint.match(/^\/profiles\/(\w+)\?/);
    if (m) return Promise.resolve({ skills: PROFILES[m[1]] ?? [] });
    return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
  }) as never);
});

describe("AnalyticsCharts (a11y)", () => {
  it("блок покрытия топ-50 - без нарушений", async () => {
    const { container } = render(<AnalyticsCharts />);
    await screen.findByText(/покрыт/i);

    await expectNoA11yViolations(container);
  });
});
