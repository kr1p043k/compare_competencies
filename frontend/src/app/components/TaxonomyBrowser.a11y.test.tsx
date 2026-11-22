/**
 * a11y: TaxonomyBrowser без нарушений axe после загрузки категорий.
 *
 * Сеть мокается на уровне lib/api.request (обёртка api() ходит через него).
 * Ждём реальные категории через findBy, прежде чем пускать axe.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { TaxonomyBrowser } from "./TaxonomyBrowser";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));

const requestMock = vi.mocked(request);

const CATEGORIES = {
  categories: [
    {
      id: "backend",
      label: "Backend",
      icon: "server",
      total: 3,
      skills: ["Python", "SQL", "PostgreSQL"],
    },
    {
      id: "data",
      label: "Данные",
      icon: "database",
      total: 2,
      skills: ["Pandas", "NumPy"],
    },
  ],
};

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u.startsWith("/api/taxonomy/categories")) return Promise.resolve(CATEGORIES);
    if (u.startsWith("/api/teacher/skills/suggestions")) {
      return Promise.resolve({ suggestions: [] });
    }
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
});

describe("TaxonomyBrowser (a11y)", () => {
  it("категории и навыки - без нарушений", async () => {
    const { container } = render(<TaxonomyBrowser />);
    await screen.findByRole("heading", { name: "Таксономия навыков" });

    await expectNoA11yViolations(container);
  });

  it("форма предложения навыка - без нарушений", async () => {
    const { container } = render(<TaxonomyBrowser showSuggest />);
    await screen.findByRole("heading", { name: "Таксономия навыков" });

    await expectNoA11yViolations(container);
  });
});
