/**
 * a11y: VacanciesList без нарушений axe (BACKLOG 6.5.3).
 *
 * Сеть мокается на уровне lib/api (request), как в VacanciesList.test.tsx:
 * fetch не вызывается, данные - минимальные (две вакансии + info).
 * Рендерим компонент в его стартовом виде и прогоняем axe по контейнеру;
 * expectNoA11yViolations() запускает axe(container) и требует пустой
 * массив violations (см. src/test/a11y.ts).
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается: тихий skip a11y-проверок запрещён.
 */
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { VacanciesList } from "./VacanciesList";

vi.mock("../../lib/api", () => ({
  download: vi.fn(),
  request: vi.fn(),
  saveBlob: vi.fn(),
}));

const requestMock = vi.mocked(request);

const ITEMS = [1, 2].map((id) => ({
  id: `v-${id}`,
  name: `Вакансия ${id}`,
  experience: "От 1 года до 3 лет",
  salary_from: 100000,
  salary_to: 150000,
  salary_currency: "RUR",
  employer_name: "Компания",
  area: "Москва",
  published_at: "2026-10-01T00:00:00",
  alternate_url: `https://hh.ru/vacancy/${id}`,
  skills: ["Python", "SQL"],
  snippet: { requirement: "Python", responsibility: "Разработка" },
}));

const INFO = {
  count: 2,
  with_skills: 2,
  file_modified: "2026-11-01 12:00",
  date_range: { from: "2026-10-01", to: "2026-11-01" },
  load_error: null,
};

function route(path: string) {
  if (path.startsWith("/api/vacancies/info")) return Promise.resolve(INFO);
  if (path.startsWith("/api/vacancies?")) {
    return Promise.resolve({
      items: ITEMS,
      total: ITEMS.length,
      limit: 12,
      offset: 0,
      has_more: false,
    });
  }
  return Promise.reject(new Error(`unexpected endpoint: ${path}`));
}

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(route as never);
});

describe("VacanciesList (a11y)", () => {
  it("список вакансий: карточки, фильтры и заголовок - без нарушений", async () => {
    const { container } = render(<VacanciesList />);
    // Дожидаемся окончания загрузки: показываем реальный список, а не
    // скелетоны (у скелетонов нет смысловой разметки для проверки).
    await screen.findByText("Вакансия 1");
    await expectNoA11yViolations(container);
  });
});
