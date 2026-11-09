import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import CompetencyTrendsPanel from "./CompetencyTrendsPanel";

// Сетевой слой подменён: тест проверяет рендер и поведение панели трендов.
vi.mock("../api", () => ({ api: vi.fn() }));

/** Ответ /competency-trends: растущая компетенция с историей навыка и стабильная без навыков. */
const TRENDS = {
  total: 2,
  trends: [
    {
      code: "ОПК-1",
      name: "Информационные технологии",
      skill_count: 2,
      active_skills_count: 1,
      change_pct: 12,
      direction: "rising",
      skills: [
        {
          name: "SQL",
          change_pct: 8,
          frequency: 40,
          prev_frequency: 20,
          history: [
            { date: "2026-01-01", freq: 10 },
            { date: "2026-02-01", freq: 40 },
          ],
        },
      ],
    },
    {
      code: "УК-2",
      name: "Коммуникация",
      skill_count: 1,
      active_skills_count: 1,
      change_pct: -2,
      direction: "stable",
      skills: [],
    },
  ],
};

beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.mocked(api).mockResolvedValue(TRENDS as never);
});

function allClassNames(): string[] {
  return Array.from(document.querySelectorAll<HTMLElement>("[class]")).flatMap((el) =>
    Array.from(el.classList),
  );
}

async function renderPanel() {
  const { container } = render(<CompetencyTrendsPanel dirCode="09.03.02" />);
  await screen.findByText("ОПК-1");
  return container;
}

describe("CompetencyTrendsPanel", () => {
  it("рисует сводку и фильтры с числами из payload", async () => {
    await renderPanel();

    expect(screen.getByText("Компетенций")).toBeInTheDocument();
    expect(screen.getByText("Растут")).toBeInTheDocument();
    expect(screen.getByText("Стабильно")).toBeInTheDocument();
    expect(screen.getByText("Падают")).toBeInTheDocument();
    expect(screen.getByText("Средний тренд")).toBeInTheDocument();
    // Средний тренд: (12 + -2) / 2 = 5.0, положительный - со знаком.
    expect(screen.getByText("+5.0%")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Все (2)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Растут (1)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Стабильно (1)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Падают (0)" })).toBeInTheDocument();
  });

  it("не подставляет HEX-цвета в className (регресс: цвета терялись в классе)", async () => {
    await renderPanel();

    // Раньше направление подставляло «#059669» прямо в className - такой
    // класс недействителен, цвет молча терялся. Токен-классы чинят это.
    const hexClasses = allClassNames().filter((cls) => cls.includes("#"));
    expect(hexClasses).toEqual([]);
  });

  it("красит процент растущей компетенции токен-классом text-success", async () => {
    await renderPanel();

    const pct = screen.getByText("↑ +12%");
    expect(pct.className).toContain("text-success");
  });

  it("раскрывает компетенцию и рисует историю навыка Sparkbars", async () => {
    const user = userEvent.setup();
    await renderPanel();

    // Клик по шапке строки раскрывает состав компетенции.
    await user.click(screen.getByText("ОПК-1"));
    expect(screen.getByText("SQL")).toBeInTheDocument();
    expect(screen.getByText("↑ +8%")).toBeInTheDocument();

    // История рисуется в обратном порядке: свежая точка слева.
    const bar2 = screen.getByTitle("2026-02-01: 40");
    const bar1 = screen.getByTitle("2026-01-01: 10");
    expect(bar2.className).toContain("bg-chart-1");
    expect(bar1.className).toContain("bg-chart-1");
    // Высоты масштабируются от максимума истории (40).
    expect((bar2 as HTMLElement).style.height).toBe("100%");
    expect((bar1 as HTMLElement).style.height).toBe("25%");
    // Подписи под столбиками - год-месяц без числа.
    expect(screen.getByText("02-01")).toBeInTheDocument();
    expect(screen.getByText("01-01")).toBeInTheDocument();

    // Стабильная компетенция без навыков честно говорит об этом.
    await user.click(screen.getByText("УК-2"));
    expect(screen.getByText("Навыки не извлечены")).toBeInTheDocument();
  });

  it("фильтр направления запрашивает срез и сужает список", async () => {
    const user = userEvent.setup();
    await renderPanel();

    await user.click(screen.getByRole("button", { name: "Растут (1)" }));
    const lastCall = vi.mocked(api).mock.calls.at(-1)?.[0];
    expect(String(lastCall)).toContain("direction=rising");
    // Стабильная УК-2 скрыта, растущая ОПК-1 осталась.
    expect(screen.getByText("ОПК-1")).toBeInTheDocument();
    expect(screen.queryByText("УК-2")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Все (2)" }));
    expect(screen.getByText("УК-2")).toBeInTheDocument();
  });
});
