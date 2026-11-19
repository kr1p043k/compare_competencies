/**
 * FaqPage: рендер, поиск, пустой результат (7.5.4). Компонент без сети.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { FaqPage } from "./FaqPage";

afterEach(cleanup);

describe("FaqPage", () => {
  it("рендерит заголовок и первый вопрос", () => {
    render(<FaqPage />);
    expect(screen.getByText("Помощь и FAQ")).toBeTruthy();
    expect(screen.getByText("Что такое Competency Gap Analyzer?")).toBeTruthy();
  });

  it("поиск сужает список вопросов", () => {
    render(<FaqPage />);
    const input = screen.getByPlaceholderText("Поиск по вопросам...");
    fireEvent.change(input, { target: { value: "Competency Gap Analyzer" } });
    expect(screen.getByText("Что такое Competency Gap Analyzer?")).toBeTruthy();
  });

  it("нет совпадений -> EmptyState", () => {
    render(<FaqPage />);
    fireEvent.change(screen.getByPlaceholderText("Поиск по вопросам..."), {
      target: { value: "zzz-нет-такого-вопроса-zzz" },
    });
    expect(screen.getByText(/Ничего не найдено по запросу/)).toBeTruthy();
  });
});
