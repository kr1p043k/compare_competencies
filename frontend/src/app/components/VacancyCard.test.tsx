import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { VacancyCard } from "./VacancyCard";

const baseVacancy = {
  id: "1",
  name: "Python Developer",
  experience: "middle",
  salary_from: 100000,
  salary_to: 150000,
  salary_currency: "RUB",
  employer_name: "Ромашка",
  employer_logo: "https://example.com/logo.png",
  area: "Москва",
  published_at: new Date().toISOString(),
  alternate_url: "https://hh.ru/vacancy/1",
  skills: ["Python", "SQL", "Docker", "Git", "Linux", "Redis", "Kafka", "Nginx", "Go"],
  snippet: { requirement: "Опыт от 3 лет, знание Python" },
};

describe("VacancyCard", () => {
  it("renders title, employer and city", () => {
    render(<VacancyCard vacancy={baseVacancy} />);
    expect(screen.getByText("Python Developer")).toBeInTheDocument();
    expect(screen.getByText("Ромашка")).toBeInTheDocument();
    expect(screen.getByText("Москва")).toBeInTheDocument();
  });

  it("formats a salary range", () => {
    render(<VacancyCard vacancy={baseVacancy} />);
    expect(screen.getByText(/100\s?000\s*-\s*150\s?000/)).toBeInTheDocument();
  });

  it("shows a placeholder when salary is missing", () => {
    render(
      <VacancyCard
        vacancy={{ ...baseVacancy, salary_from: undefined, salary_to: undefined }}
      />,
    );
    expect(screen.getByText(/не указана/i)).toBeInTheDocument();
  });

  it("caps visible skills at seven and shows the remainder", () => {
    render(<VacancyCard vacancy={baseVacancy} />);
    expect(screen.getByText("Python")).toBeInTheDocument();
    expect(screen.getByText("Kafka")).toBeInTheDocument();
    expect(screen.queryByText("Nginx")).not.toBeInTheDocument();
    expect(screen.queryByText("Go")).not.toBeInTheDocument();
    expect(screen.getByText("+2")).toBeInTheDocument();
  });

  it("uses the employer logo when present and a placeholder otherwise", () => {
    const { container, rerender } = render(<VacancyCard vacancy={baseVacancy} />);
    expect(container.querySelector("img")).toHaveAttribute("src", "https://example.com/logo.png");

    rerender(<VacancyCard vacancy={{ ...baseVacancy, employer_logo: undefined }} />);
    expect(container.querySelector("img")).toBeNull();
  });

  it("renders the list variant as a compact row", () => {
    const { container } = render(<VacancyCard vacancy={baseVacancy} variant="list" />);
    // В list-варианте карточка строчная
    expect(container.querySelector('[data-slot="card"]')).toHaveClass("flex-row");
  });

  it("exposes the hh.ru link", () => {
    render(<VacancyCard vacancy={baseVacancy} />);
    expect(screen.getByTitle("Открыть на hh.ru")).toHaveAttribute(
      "href",
      "https://hh.ru/vacancy/1",
    );
  });

  it("marks spam vacancies", () => {
    render(<VacancyCard vacancy={{ ...baseVacancy, is_spam: true, spam_reason: "дубль" }} />);
    expect(screen.getByText("Спам")).toBeInTheDocument();
  });
});