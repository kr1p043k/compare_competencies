/**
 * Самопроверка детектора: заведомо недоступная разметка ОБЯЗАНА давать
 * нарушение. Если этот тест падает - expectNoA11yViolations перестал
 * ловить проблемы (например, правила отключили слишком широко), и все
 * «зелёные» a11y-тесты перестали что-либо гарантировать.
 *
 * Без установленного jest-axe тест тоже красный: ошибка «не установлен»
 * не содержит id правила, поэтому настоящее срабатывание детектора
 * отличается от отсутствия пакета.
 */
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { expectNoA11yViolations } from "./a11y";

describe("expectNoA11yViolations (selfcheck)", () => {
  it("ловит изображение без alt (правило image-alt)", async () => {
    const { container } = render(<img src="https://example.com/x.png" />);
    await expect(expectNoA11yViolations(container)).rejects.toThrowError(
      /image-alt/,
    );
  });
});
