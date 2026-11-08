/**
 * a11y: VacancyCard без нарушений axe в обоих вариантах вёрстки (grid/list).
 *
 * Компонент чисто презентационный: данные приходят пропсом, сеть и роутер
 * не задействованы. Дата публикации зафиксирована в прошлом: formatDate
 * ветвится на «сегодня/вчера» через сравнение с текущей датой, фиксики
 * дают детерминированный вывод.
 *
 * Без установленного jest-axe тесты падают с подсказкой по установке,
 * а не пропускаются (см. src/test/a11y.ts).
 */
import { render } from "@testing-library/react";
import { describe, it } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { VacancyCard } from "./VacancyCard";

const vacancy = {
  id: "1",
  name: "Python Developer",
  experience: "middle" as const,
  salary_from: 100000,
  salary_to: 150000,
  salary_currency: "RUB",
  employer_name: "Ромашка",
  employer_logo: "https://example.com/logo.png",
  area: "Москва",
  // Фиксированная дата в прошлом: diffDays всегда > 7, карточка никогда
  // не прыгает между «сегодня»/«вчера» на границе суток.
  published_at: "2024-06-15T09:00:00Z",
  alternate_url: "https://hh.ru/vacancy/1",
  skills: ["Python", "SQL", "Docker", "Git", "Linux", "Redis", "Kafka", "Nginx", "Go"],
  snippet: { requirement: "Опыт от 3 лет, знание Python" },
};

describe("VacancyCard (a11y)", () => {
  it("grid-вариант: логотип, бейдж уровня, скиллы, зарплата - без нарушений", async () => {
    const { container } = render(<VacancyCard vacancy={vacancy} />);
    await expectNoA11yViolations(container);
  });

  it("list-вариант: компактная строка - без нарушений", async () => {
    const { container } = render(<VacancyCard vacancy={vacancy} variant="list" />);
    await expectNoA11yViolations(container);
  });

  it("без логотипа и зарплаты (placeholder-ветки) - без нарушений", async () => {
    const { container } = render(
      <VacancyCard
        vacancy={{
          ...vacancy,
          employer_logo: undefined,
          salary_from: undefined,
          salary_to: undefined,
          skills: [],
        }}
      />,
    );
    await expectNoA11yViolations(container);
  });
});
