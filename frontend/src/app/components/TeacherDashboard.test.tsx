import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { TeacherDashboard } from "./TeacherDashboard";

// Сетевой слой подменён целиком: TeacherDashboard на маунте тянет агрегаты
// teacher-анализа, тест проверяет рендер payload, а не HTTP.
vi.mock("../api", () => ({
  api: vi.fn(() => Promise.reject(new Error("api mock not configured"))),
}));

// Панель чата подменена заглушкой: тест проверяет СВЯЗЬ дашборда с панелью
// (какой ключ контекста он передаёт), а не рендер самой панели.
const { chatProps } = vi.hoisted(() => ({ chatProps: vi.fn() }));
vi.mock("./LlmChatPanel", () => ({
  LlmChatPanel: (props: Record<string, unknown>) => {
    chatProps(props);
    return null;
  },
}));

// reportError уходит в /api/log реальным fetch - в jsdom он не нужен.
vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(() => ""),
  onError: vi.fn(() => () => {}),
}));

/** Ответ /teacher/krm/directions: одно направление. */
const DIRECTIONS = [
  { dir_code: "09.03.02", name: "Информационные системы", profile: "", disciplines_count: 2 },
];

/** Ответ /teacher/krm/disciplines: список слева. */
const DISCIPLINES = [
  { id: "d1", name: "Базы данных", in_scope: true, competencies_count: 3, skills_count: 12 },
  { id: "d2", name: "Web-разработка", in_scope: true, competencies_count: 2, skills_count: 9 },
];

/** Ответ /teacher/krm/disciplines/<имя>: деталка направления без dir_code. */
const DISCIPLINE_DETAIL = {
  id: "d1",
  name: "Базы данных",
  competencies: [],
};

/**
 * Ответ /teacher/analysis: срез направления с парами метрик.
 * weighted_coverage есть только у «Базы данных» - вторая дисциплина
 * проверяет, что пара «кач.» не рисуется из ничего.
 */
const ANALYSIS = {
  direction: "09.03.02",
  direction_name: "Информационные системы",
  profile: "",
  total_disciplines: 2,
  average_coverage: 0.9,
  average_market_top_coverage: 0.62,
  average_market_top_reliable: true,
  average_market_weight_coverage: 0.48,
  coverage_level: "medium",
  total_gaps_across_all: 9,
  top_cross_discipline_gaps: [
    { skill: "OceanBase", disciplines: 3 },
    { skill: "Berkeley DBXML", disciplines: 1 },
  ],
  top_emerging_across_all: [
    { skill: "Ray ML", frequency: 3, change_pct: 12.7, basis: "рост спроса" },
    { skill: "k6", frequency: 4 },
  ],
  recommendations: [],
  trends: { rising: [], declining: [] },
  disciplines: [
    {
      name: "Базы данных",
      coverage_ratio: 0.55,
      weighted_coverage: 0.4,
      coverage_level: "medium",
      gaps: 2,
      emerging: 1,
      market_top_coverage: 0.6,
      market_weight_coverage: 0.5,
      market_top_total: 10,
      market_top_reliable: true,
      market_top_source: "explicit",
      years: [2025],
    },
    {
      name: "Web-разработка",
      coverage_ratio: 0.3,
      coverage_level: "low",
      gaps: 4,
      emerging: 0,
      years: [2025],
    },
  ],
  generated_at: "2026-01-01T00:00:00Z",
};

function route(endpoint: string): Promise<unknown> {
  // meta проверяем раньше анализа: у обоих общий префикс /teacher/analysis.
  if (endpoint.startsWith("/teacher/analysis/meta")) return Promise.resolve({ stale: false });
  if (endpoint.startsWith("/teacher/analysis")) return Promise.resolve(ANALYSIS);
  if (endpoint.startsWith("/teacher/krm/recommendations")) return Promise.resolve([]);
  if (endpoint.startsWith("/teacher/krm/directions")) return Promise.resolve(DIRECTIONS);
  // деталка дисциплины: общий префикс со списком, поэтому проверка раньше.
  if (endpoint.startsWith("/teacher/krm/disciplines/")) return Promise.resolve(DISCIPLINE_DETAIL);
  if (endpoint.startsWith("/teacher/krm/disciplines")) return Promise.resolve(DISCIPLINES);
  if (endpoint.startsWith("/teacher/krm/scope-summary")) {
    return Promise.resolve({
      total: 2,
      in_scope: 2,
      excluded_manual: 0,
      excluded_no_data: 0,
      analyzed: 2,
    });
  }
  if (endpoint.startsWith("/teacher/krm/stats")) {
    return Promise.resolve({
      total_disciplines: 2,
      total_competencies: 5,
      total_skills: 30,
      total_reports: 2,
    });
  }
  if (endpoint.startsWith("/teacher/rpd/sources")) return Promise.resolve({ yandex_covered: [] });
  return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
}

beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.mocked(api).mockImplementation(route as never);
});

/** Рендер + ожидание данных анализа (приходят последним эффектом). */
async function renderDashboard() {
  render(<TeacherDashboard />);
  await screen.findByText("по спросу: 48.0%");
}

describe("TeacherDashboard", () => {
  it("показывает строгий охват рынка с парой «по спросу» в сводке", async () => {
    await renderDashboard();
    // Строгий охват - в двух местах: сводка слева и KPI направления.
    expect(screen.getAllByText("62.0%")).toHaveLength(2);
    expect(screen.getByText("Средний охват рынка")).toBeInTheDocument();
    // Пара есть и в KPI, со своим пояснением.
    expect(screen.getByText("взвешенный охват: 48.0%")).toBeInTheDocument();
    expect(
      screen.getByTitle(
        "Взвешенный охват (по спросу): насколько востребовано то из топа рынка, что реально покрыто. Может быть выше или ниже строгого",
      ),
    ).toBeInTheDocument();
    // Происхождение знаменателя: explicit у одной дисциплины, у второй
    // поля нет - она честно показана как n/a. В шаговом рефакторе (main)
    // это перенесено в tooltip карточки сводки (title), а не видимый текст.
    expect(screen.getByTitle("источник рынка - explicit: 1, n/a: 1")).toBeInTheDocument();
    // Знаменатель достоверен: предупреждения нет.
    expect(screen.queryByText(/недостоверно/)).not.toBeInTheDocument();
  });

  it("рисует пару «кач.» в строке дисциплины только из payload", async () => {
    await renderDashboard();
    // У «Базы данных» weighted_coverage есть: строгий + качество рядом.
    expect(screen.getByText("55%")).toBeInTheDocument();
    expect(screen.getByText("кач.: 40%")).toBeInTheDocument();
    // У «Web-разработка» поля нет: пара не появилась, весь дашборд держит
    // ровно одну подпись «кач.».
    expect(screen.getByText("30%")).toBeInTheDocument();
    expect(screen.getAllByText(/кач\./)).toHaveLength(1);
  });

  it("выводит разрывы РПД и emerging-навыки с происхождением", async () => {
    await renderDashboard();
    // Кнопка суб-навигации и якорь раздела.
    expect(screen.getByRole("button", { name: "Разрывы" })).toBeInTheDocument();
    expect(document.getElementById("t-gaps")).not.toBeNull();

    expect(screen.getByText("Разрывы между планом и рынком")).toBeInTheDocument();
    expect(screen.getByText("OceanBase")).toBeInTheDocument();
    expect(screen.getByText("3 дисциплины")).toBeInTheDocument();
    expect(screen.getByText("1 дисциплина")).toBeInTheDocument();
    // Тултип объясняет, откуда взялся чип разрыва.
    expect(
      screen.getByTitle("Навык есть в планах 3 дисциплинах, но в свежих вакансиях спроса нет"),
    ).toBeInTheDocument();

    // Emerging: частота вакансий у обоих, динамика - только где она есть.
    expect(screen.getByText("Ray ML")).toBeInTheDocument();
    expect(screen.getByText("3 вк.")).toBeInTheDocument();
    expect(screen.getByText("+13%")).toBeInTheDocument();
    expect(screen.getAllByText(/^\+\d+%$/)).toHaveLength(1);
    expect(screen.getByText("k6")).toBeInTheDocument();
    expect(screen.getByText("4 вк.")).toBeInTheDocument();
    expect(
      screen.getByTitle("Вакансий с этим навыком: 3 · основа отбора: рост спроса"),
    ).toBeInTheDocument();
  });

  it("различает набор (перезагружает данные) и фильтр видимости списка", async () => {
    const user = userEvent.setup();
    await renderDashboard();

    // Оба контроля названы по роли: «Набор» грузит данные, «Показать в
    // списке» лишь скрывает строки. Тултипы проговаривают разницу.
    expect(screen.getByText("Набор:")).toBeInTheDocument();
    expect(screen.getByTitle(/запрашивает данные с сервера/)).toBeInTheDocument();
    expect(screen.getByText(/Показать в списке:/)).toBeInTheDocument();
    expect(screen.getByTitle(/Данные не перезагружает/)).toBeInTheDocument();

    // Фильтр видимости: у обеих дисциплин из мока years=[2025],
    // поэтому «в плане 2024» прячет строки и НЕ обращается к API.
    const callsBefore = vi.mocked(api).mock.calls.length;
    await user.click(screen.getByRole("button", { name: "в плане 2024" }));
    expect(screen.queryByRole("button", { name: /Базы данных/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Web-разработка/ })).not.toBeInTheDocument();
    expect(vi.mocked(api).mock.calls.length).toBe(callsBefore);

    // Набор: то же «2024» здесь - новое source of truth, анализ
    // запрашивается заново с year=2024.
    await user.click(screen.getByRole("button", { name: "2024" }));
    await waitFor(() => {
      const yearCalls = vi.mocked(api).mock.calls.filter(([p]) => String(p).includes("year=2024"));
      expect(yearCalls.length).toBeGreaterThan(0);
    });

    // Фильтр не сбросился сам: вернуть строки - кнопка «все».
    await user.click(screen.getByRole("button", { name: "все" }));
    expect(screen.getByRole("button", { name: /Базы данных/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Web-разработка/ })).toBeInTheDocument();
  });

  it("передаёт чату код направления, а не имя дисциплины", async () => {
    const user = userEvent.setup();
    await renderDashboard();

    // До выбора дисциплины: ключ контекста - уже код направления из KRM
    // (selectedDir), а не null и не человекочитаемое имя.
    expect(chatProps).toHaveBeenLastCalledWith(
      expect.objectContaining({ role: "teacher", discipline: "09.03.02" }),
    );

    // Выбор дисциплины («Базы данных») меняет подпись (contextLabel),
    // но НЕ ключ: бэкенд llm_chat читает teacher/<код>/_summary.json и
    // krm_disciplines_<код>.json, имя давало пустой контекст.
    await user.click(screen.getByRole("button", { name: /Базы данных/ }));
    await waitFor(() =>
      expect(chatProps).toHaveBeenLastCalledWith(
        expect.objectContaining({
          role: "teacher",
          discipline: "09.03.02",
          contextLabel: "Базы данных",
        }),
      ),
    );

    // Ни в одном рендере ключом не было человеческого имени/названия.
    for (const [props] of chatProps.mock.calls) {
      expect(props.discipline).not.toBe("Базы данных");
      expect(props.discipline).not.toBe("Информационные системы");
    }
  });
});
