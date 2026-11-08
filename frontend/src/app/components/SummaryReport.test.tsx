import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { SummaryReport } from "./SummaryReport";

/** gaps = asdict(SkillMetrics): полная запись с происхождением. */
const fullGap = {
  skill: "Docker",
  gap_j: 1,
  gap_m: 0.8,
  gap_s: 0.6,
  demand_j: 0.7,
  demand_m: 0.75,
  demand_s: 0.8,
  user_level: 0.1,
  cluster_relevance: 0.66,
  importance: 0.8,
  category: "missing",
};

/** Минимальный gap старого формата: только то, что сериализовалось раньше. */
const slimGap = {
  skill: "Windows Server",
  gap_j: 0.9,
  importance: 0.4,
  category: "missing",
};

const data = {
  profiles: ["Python-разработчик", "Go-разработчик"],
  evaluations: {
    "Python-разработчик": {
      target_profession: "Data Engineer",
      dominant_domain_name: "Backend",
      market_coverage_score: 46.4,
      skill_coverage: 58.9,
      readiness_score: 41.2,
      domain_coverage_score: 33.3,
      market_skill_coverage: 28.1,
      coverage_weighted: 58.9,
      coverage_strict_scope: "топ-200 рынка",
      avg_gap: 54.9,
      closest_roles: [{ role: "Data Engineer", semantic_similarity: 62.5 }],
      gaps: { docker: fullGap, "windows-admin": slimGap },
    },
    "Go-разработчик": {
      target_profession: "Backend-разработчик",
      market_coverage_score: 51.1,
      gaps: {},
    },
  },
};

async function openProfile(name: string) {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: new RegExp(name) }));
}

describe("SummaryReport", () => {
  it("показывает пару рынок + взвешенное в хедере профиля", () => {
    render(<SummaryReport data={data} />);
    // market_coverage_score виден дважды by design: таблица «Сравнение
    // профилей» и MetricPair в хедере карточки.
    expect(screen.getAllByText("46.4%")).toHaveLength(2);
    expect(screen.getByText("взвеш.: 58.9%")).toBeInTheDocument();
  });

  it("не выводит «взвеш.», когда skill_coverage нет в payload", () => {
    render(<SummaryReport data={data} />);
    // Таблица + хедер карточки (см. предыдущий тест).
    expect(screen.getAllByText("51.1%")).toHaveLength(2);
    // У Go-разработчика пары быть не должно. Проверяем точную строку пары:
    // в таблице есть label «Покрытие навыков (взвеш.)» и hint с «взвеш.»
    // выше, но это не пара MetricPair.
    expect(screen.getAllByText("взвеш.: 58.9%")).toHaveLength(1);
  });

  it("gap с полными полями получает drill-down «откуда взято», короткий - нет", async () => {
    render(<SummaryReport data={data} />);
    await openProfile("Python-разработчик");

    // Полный gap: drill-down есть, внутри - происхождение по уровням.
    const originSummaries = screen.getAllByText("откуда взято");
    expect(originSummaries).toHaveLength(1);

    const user = userEvent.setup();
    await user.click(originSummaries[0]);
    // Человекочитаемые подписи вместо сырых ключей gap_m / спрос J / ...
    // разрыв Junior не дублируем: он уже в заголовке строки («+100%»).
    expect(screen.getByText("Разрыв (M/S)")).toBeInTheDocument();
    expect(screen.getByText("Спрос (J/M/S)")).toBeInTheDocument();
    expect(screen.getByText("Ваш уровень")).toBeInTheDocument();
    expect(screen.getByText("Кластер")).toBeInTheDocument();
    // Те же значения, что и раньше; доступное имя бара = tooltip «что это».
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Middle: насколько не хватает до требуемого" }),
    ).toHaveTextContent("80%");
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Senior: насколько не хватает до требуемого" }),
    ).toHaveTextContent("60%");
    expect(
      screen.getByRole("meter", { name: "Востребованность на рынке, уровень Junior" }),
    ).toHaveTextContent("70%");
    expect(
      screen.getByRole("meter", { name: "Востребованность на рынке, уровень Middle" }),
    ).toHaveTextContent("75%");
    expect(
      screen.getByRole("meter", { name: "Востребованность на рынке, уровень Senior" }),
    ).toHaveTextContent("80%");
    expect(
      screen.getByRole("meter", { name: "Ваш текущий уровень владения навыком" }),
    ).toHaveTextContent("10%");
    expect(
      screen.getByRole("meter", { name: "Релевантность навыка кластеру целевых вакансий" }),
    ).toHaveTextContent("66%");

    // Короткий gap без происхождения: строки GapRow на месте, «откуда взято» один.
    expect(screen.getByText("Windows Server")).toBeInTheDocument();
    expect(screen.getAllByText("откуда взято")).toHaveLength(1);
  });

  it("таблица метрик рисует строки только по пришедшим полям", () => {
    render(<SummaryReport data={data} />);
    expect(screen.getByText("Покрытие рынка")).toBeInTheDocument();
    expect(screen.getByText("Средний разрыв")).toBeInTheDocument();
    // profession_coverage отсутствует в payload - строка скрыта.
    expect(screen.queryByText("Покрытие профессии")).not.toBeInTheDocument();
  });

  it("категория и важность видны в строках разрывов", async () => {
    render(<SummaryReport data={data} />);
    await openProfile("Python-разработчик");
    // Полный и короткий gap показывают свои важности и разрывы Junior.
    expect(screen.getByText("важность 80%")).toBeInTheDocument();
    expect(screen.getByText("важность 40%")).toBeInTheDocument();
    expect(screen.getByText("+100%")).toBeInTheDocument();
    expect(screen.getByText("+90%")).toBeInTheDocument();
    // Категория у обоих missing.
    expect(screen.getAllByText("missing")).toHaveLength(2);
  });
});
