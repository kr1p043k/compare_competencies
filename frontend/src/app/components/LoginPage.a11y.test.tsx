/**
 * a11y: LoginPage без нарушений axe в начальном состоянии (экран входа).
 *
 * На монтирование компонент сети не зовёт: request уходит только из
 * submit-обработчиков, тест форму не отправляет. Роутер не используется.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render } from "@testing-library/react";
import { describe, it } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { LoginPage } from "./LoginPage";

describe("LoginPage (a11y)", () => {
  it("экран входа (подписанные поля, кнопка) без нарушений", async () => {
    const { container } = render(<LoginPage onLogin={() => {}} />);
    await expectNoA11yViolations(container);
  });
});
