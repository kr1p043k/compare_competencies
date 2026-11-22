/**
 * a11y: FaqPage без нарушений axe. Компонент без сети, данные статичны.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { FaqPage } from "./FaqPage";

describe("FaqPage (a11y)", () => {
  it("полный список вопросов - без нарушений", async () => {
    const { container } = render(<FaqPage />);
    await screen.findByText("Что такое Competency Gap Analyzer?");

    await expectNoA11yViolations(container);
  });

  it("состояние «ничего не найдено» - без нарушений", async () => {
    const user = userEvent.setup();
    const { container } = render(<FaqPage />);

    await user.type(
      screen.getByPlaceholderText("Поиск по вопросам..."),
      "zzz-нет-такого-вопроса-zzz",
    );
    await screen.findByText(/Ничего не найдено по запросу/);

    await expectNoA11yViolations(container);
  });
});
