/**
 * a11y: DataTab без нарушений axe после завершения загрузки результата.
 *
 * Компонент чисто презентационный: все данные и обработчики приходят
 * пропсами, сети и роутера нет (подстраховочно мокаем lib/api.request).
 * Сначала рендерим состояние загрузки, затем переключаем на готовый
 * результат и ждём EmptyState через findBy, прежде чем пускать axe.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import { describe, it, vi } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { DataTab } from "./DataTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));

const baseProps = {
  profile: "backend",
  profilesList: ["backend", "frontend"],
  onProfileChange: () => {},
  loading: false,
  isStudent: false,
  gapRunning: false,
  gapMsg: "",
  onLoadRecommendations: () => {},
  onLoadSummary: () => {},
  onRunGapAnalysis: () => {},
  professionsList: ["Аналитик", "Разработчик"],
  targetProfession: "",
  onTargetProfessionChange: () => {},
  onCompareWithProfession: () => {},
  resultLoadedAt: {},
  lastResult: null,
  onDownloadPdf: () => {},
};

describe("DataTab (a11y)", () => {
  it("после загрузки (EmptyState) - без нарушений", async () => {
    const { container, rerender } = render(<DataTab {...baseProps} loading />);
    await screen.findByRole("status", { name: "Загрузка результата" });

    rerender(<DataTab {...baseProps} loading={false} />);
    await screen.findByText("Результатов пока нет");

    await expectNoA11yViolations(container);
  });
});
