/**
 * a11y: StudentKrm без нарушений axe после загрузки программы КРМ.
 *
 * Сеть мокается на уровне lib/api.request по образцу
 * StudentKrm.coverage.test.tsx; fetch подменён заглушкой, чтобы reportError
 * не ходил в /api/log.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { StudentKrm } from "./StudentKrm";

vi.mock("../../lib/api", () => {
  class ApiError extends Error {
    constructor(
      message: string,
      readonly status = 0,
      readonly detail: unknown = null,
      readonly url = "",
    ) {
      super(message);
      this.name = "ApiError";
    }
  }
  return { request: vi.fn(), ApiError };
});

const requestMock = vi.mocked(request);

const DIRECTIONS = {
  directions: [{ dir_code: "09.03.02", name: "Информатика и вычислительная техника" }],
};

const PROFILES = { profiles: ["base", "dc", "top_dc"] };

const KRM = {
  direction: "09.03.02",
  direction_name: "Информатика и вычислительная техника",
  profile: "base",
  disciplines: [
    {
      discipline: "Базы данных",
      competencies: [
        {
          code: "ПК-1",
          skills: [
            { text: "SQL", student_has: true },
            { text: "Индексы", student_has: false },
          ],
        },
      ],
    },
    {
      discipline: "Сети",
      competencies: [{ code: "ПК-2", skills: [] }],
    },
  ],
  merged: [],
  counts: { krm: 2, student: 1, overlap: 1, merged: 0, student_only: 0 },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u === "/api/krm/directions") return Promise.resolve(DIRECTIONS);
    if (u === "/api/profiles") return Promise.resolve(PROFILES);
    if (u.startsWith("/api/krm/student/competencies")) return Promise.resolve(KRM);
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response("{}", { status: 200 }))),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("StudentKrm (a11y)", () => {
  it("загруженная программа и фильтры - без нарушений", async () => {
    const user = userEvent.setup();
    const { container } = render(<StudentKrm />);
    await user.click(screen.getByRole("button", { name: "Показать" }));
    await screen.findByText("Базы данных");

    await expectNoA11yViolations(container);
  });
});
