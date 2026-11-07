/**
 * a11y: PageHeader без нарушений axe (базовый контракт доступности).
 *
 * Компонент чисто презентационный: без сети и роутера. Без установленного
 * jest-axe тесты падают с подсказкой по установке, а не пропускаются
 * (см. src/test/a11y.ts).
 */
import { render } from "@testing-library/react";
import { describe, it } from "vitest";
import { Briefcase } from "lucide-react";

import { expectNoA11yViolations } from "../../test/a11y";
import { PageHeader } from "./PageHeader";

describe("PageHeader (a11y)", () => {
  it("максимальная сборка (иконка, описание, действия) без нарушений", async () => {
    const { container } = render(
      <PageHeader
        title="Вакансии"
        description="Список вакансий"
        icon={Briefcase}
        actions={<button type="button">Обновить</button>}
      />,
    );
    await expectNoA11yViolations(container);
  });

  it("минимальная сборка (только заголовок) без нарушений", async () => {
    const { container } = render(<PageHeader title="Аналитика" />);
    await expectNoA11yViolations(container);
  });
});
