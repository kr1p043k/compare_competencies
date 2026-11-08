import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { ArticlesPage } from "./ArticlesPage";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("./AnalyticsCharts", () => ({
  AnalyticsCharts: () => <div data-testid="analytics-charts-stub" />,
}));

const apiMock = vi.mocked(api);

const STATS = {
  total: 120,
  by_experience: { junior: 10, middle: 20, senior: 30 },
  salary: { average: 150000, min: 30000, max: 400000, count: 60 },
};

const ANALYTICS = {
  salary_by_level: { junior: { count: 10, average: 90000 } },
  top_regions: [{ name: "Москва", count: 12 }],
  top_employers: [{ name: "Яндекс", count: 5 }],
  skills: { with_skills: 80, total: 120, percent: 66, avg_skills_per_vacancy: 5 },
};

function coverageResponse(n: number) {
  const coverage: Record<
    string,
    { label: string; icon: string; total: number; covered: number; percent: number }
  > = {};
  for (let i = 1; i <= n; i++) {
    coverage[`cat${i}`] = {
      label: `Категория ${i}`,
      icon: "",
      total: 100,
      covered: i,
      percent: i * 10,
    };
  }
  return { coverage };
}

const PROFESSIONS = [
  {
    name: "Аналитик данных",
    snapshot: "freq_profession_Аналитик данных_2026-11.json",
    date: "2026-11-02",
  },
  {
    name: "Backend-разработчик",
    snapshot: "freq_profession_Backend-разработчик_2026-11.json",
    date: "",
  },
];

function byProfession(profession: string) {
  if (profession === "Аналитик данных") {
    return {
      profession,
      source: "snapshot",
      snapshot_date: "2026-11-02",
      skills: [
        { skill: "SQL", frequency: 40 },
        { skill: "Python", frequency: 30 },
      ],
    };
  }
  return {
    profession,
    source: "live",
    skills: [{ skill: "Docker", frequency: 7 }],
  };
}

function route(professions: typeof PROFESSIONS | []) {
  return (endpoint: string) => {
    // Порядок важен: /trends/professions и /trends/by-profession сравниваются
    // раньше «голого» /trends, иначе первый префикс поглотит их.
    if (endpoint.startsWith("/trends/professions")) return Promise.resolve({ professions });
    if (endpoint.startsWith("/trends/by-profession")) {
      const prof =
        new URLSearchParams(endpoint.split("?")[1] ?? "").get("profession") ?? "";
      return Promise.resolve(byProfession(prof));
    }
    if (endpoint.startsWith("/trends")) {
      return Promise.resolve({ trends: { rising: [], falling: [] } });
    }
    if (endpoint.startsWith("/vacancies/stats/summary")) return Promise.resolve(STATS);
    if (endpoint.startsWith("/vacancies/stats/analytics")) return Promise.resolve(ANALYTICS);
    if (endpoint.startsWith("/market/top-skills")) return Promise.resolve({ skills: [] });
    if (endpoint.startsWith("/taxonomy/coverage")) return Promise.resolve(coverageResponse(7));
    return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
  };
}

beforeEach(() => {
  apiMock.mockReset();
});

describe("ArticlesPage. Тренды по профессиям", () => {
  it("при пустом списке показывает короткий empty-state без служебных команд", async () => {
    apiMock.mockImplementation(route([]) as never);
    render(<ArticlesPage />);

    expect(
      await screen.findByText("Данные по профессиям появятся после запуска анализа.")
    ).toBeInTheDocument();
    // Служебные детали (CLI-команда, имена файлов) не показываются пользователю.
    expect(screen.queryByText(/python/i)).toBeNull();
    expect(screen.queryByText(/freq_profession/)).toBeNull();
    expect(screen.queryByText(/snapshot_professions/)).toBeNull();
    expect(screen.queryByText(/не чаще раза в сутки/)).toBeNull();
    // Селектора нет: выбирать не из чего.
    expect(screen.queryByRole("combobox")).toBeNull();
  });

  it("тянет список снапшотов: в пунктах имя профессии и дата снапшота", async () => {
    const user = userEvent.setup();
    apiMock.mockImplementation(route(PROFESSIONS) as never);
    render(<ArticlesPage />);

    const trigger = await screen.findByRole("combobox", { name: "Профессия для трендов" });
    await user.click(trigger);

    expect(
      await screen.findByRole("option", { name: "Аналитик данных · 2026-11-02" })
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Backend-разработчик" })).toBeInTheDocument();
  });

  it("выбор профессии грузит навыки и подпись источника-снапшота", async () => {
    const user = userEvent.setup();
    apiMock.mockImplementation(route(PROFESSIONS) as never);
    render(<ArticlesPage />);

    const trigger = await screen.findByRole("combobox", { name: "Профессия для трендов" });
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Аналитик данных · 2026-11-02" }));

    expect(await screen.findByText("SQL")).toBeInTheDocument();
    expect(screen.getByText("Снапшот от 2026-11-02")).toBeInTheDocument();

    const byProfCall = apiMock.mock.calls
      .map(([endpoint]) => String(endpoint))
      .find((endpoint) => endpoint.startsWith("/trends/by-profession"));
    expect(byProfCall).toBeDefined();
    expect(byProfCall).toContain(encodeURIComponent("Аналитик данных"));
  });

  it("без снапшота честно помечает источник как живой расчёт по базе", async () => {
    const user = userEvent.setup();
    apiMock.mockImplementation(route(PROFESSIONS) as never);
    render(<ArticlesPage />);

    const trigger = await screen.findByRole("combobox", { name: "Профессия для трендов" });
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "Backend-разработчик" }));

    expect(await screen.findByText("Docker")).toBeInTheDocument();
    expect(
      screen.getByText("Живой расчёт по базе: снапшота для этой профессии ещё нет.")
    ).toBeInTheDocument();
  });
});

describe("ArticlesPage. Покрытие таксономии навыков", () => {
  it("свернуто до 5 категорий, разворачивается кнопкой", async () => {
    const user = userEvent.setup();
    apiMock.mockImplementation(route([]) as never);
    render(<ArticlesPage />);

    // Сортировка по percent desc: сверху «Категория 7» (70%), ниже по убыванию.
    expect(await screen.findByText("Категория 7")).toBeInTheDocument();
    expect(screen.getByText("Категория 3")).toBeInTheDocument();
    expect(screen.queryByText("Категория 2")).toBeNull();
    expect(screen.queryByText("Категория 1")).toBeNull();

    await user.click(screen.getByRole("button", { name: /Показать ещё 2 из 7/ }));

    expect(await screen.findByText("Категория 1")).toBeInTheDocument();
    expect(screen.getByText("Категория 2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Свернуть/ })).toBeInTheDocument();
  });
});
