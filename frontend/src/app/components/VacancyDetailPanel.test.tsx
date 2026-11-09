/**
 * VacancyDetailPanel: загрузка деталей вакансии + секции.
 * Сеть мокается на уровне lib/api (request); регресс-щит на рефактор панели.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { VacancyDetailPanel } from "./VacancyDetailPanel";

vi.mock("../../lib/api", () => ({
  download: vi.fn(),
  request: vi.fn(),
  saveBlob: vi.fn(),
}));
vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

afterEach(cleanup);

const requestMock = vi.mocked(request);

const SUMMARY = { id: "v-1", name: "Старший Fullstack", employer_name: "Компания" };
const DETAIL = {
  id: "v-1",
  name: "Старший Fullstack (детально)",
  alternate_url: "https://hh.ru/vacancy/v-1",
  key_skills: [{ name: "Python" }, { name: "SQL" }],
  skills: ["FastAPI"],
  description: "Опыт коммерческой разработки от трёх лет.",
  employer: { name: "Компания" },
  area: { name: "Москва" },
};

describe("VacancyDetailPanel", () => {
  it("показывает состояние загрузки и запрашивает детали один раз", async () => {
    let resolve!: (v: unknown) => void;
    requestMock.mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }) as never,
    );

    render(<VacancyDetailPanel vacancy={SUMMARY} onClose={() => {}} />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith("/api/vacancies/v-1");

    resolve(DETAIL);
    await waitFor(() =>
      expect(screen.queryByRole("status")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Старший Fullstack (детально)")).toBeInTheDocument();
  });

  it("рисует загруженные секции: заголовок, ключевые навыки и ссылку hh.ru", async () => {
    requestMock.mockResolvedValue(DETAIL as never);

    render(<VacancyDetailPanel vacancy={SUMMARY} onClose={() => {}} />);

    const heading = await screen.findByRole("heading", {
      level: 4,
      name: /Старший Fullstack/,
    });
    expect(heading).toBeInTheDocument();
    expect(requestMock).toHaveBeenCalledWith("/api/vacancies/v-1");

    expect(screen.getByText("Ключевые навыки (HH)")).toBeInTheDocument();
    expect(screen.getByText("Python")).toBeInTheDocument();
    expect(screen.getByText("SQL")).toBeInTheDocument();

    const link = screen.getByRole("link", { name: /Открыть на hh\.ru/ });
    expect(link).toHaveAttribute("href", "https://hh.ru/vacancy/v-1");
    expect(link).toHaveAttribute("target", "_blank");
  });
});