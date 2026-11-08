import { describe, expect, it } from "vitest";

import {
  competencyFilterCounts,
  filterCompetencies,
  filterDisciplines,
  filterRecommendations,
  disciplineFilterCounts,
  ksaCount,
  sortDisciplines,
} from "./teacherFilters";

const analysis = new Map<string, { coverage_ratio: number; coverage_level: string; gaps: number; years?: number[] }>([
  ["Алгоритмы", { coverage_ratio: 0.8, coverage_level: "good", gaps: 2, years: [2025] }],
  ["Базы данных", { coverage_ratio: 0.3, coverage_level: "mid", gaps: 0, years: [2024, 2025] }],
  ["Физкультура", { coverage_ratio: 0, coverage_level: "no_data", gaps: 0, years: [2024] }],
]);

const discs = [
  { name: "Алгоритмы" },
  { name: "Базы данных" },
  { name: "Физкультура" },
];

describe("ksaCount", () => {
  it("считает по группам ЗУН, а при их отсутствии — по skills", () => {
    expect(ksaCount({ id: "1", code: "ПК-1", ksa: { knowledge: [{}, {}], abilities: [], skills: [{}] } })).toBe(3);
    expect(ksaCount({ id: "2", code: "ПК-2", skills: ["a", "b"] })).toBe(2);
    expect(ksaCount({ id: "3", code: "ПК-3" })).toBe(0);
    // пустые группы + fallback skills: берём skills
    expect(ksaCount({ id: "4", code: "ПК-4", skills: ["x"], ksa: { knowledge: [], abilities: [], skills: [] } })).toBe(1);
  });
});

describe("filterDisciplines", () => {
  it("пропускает всё без фильтров", () => {
    expect(filterDisciplines(discs, analysis, {})).toHaveLength(3);
  });

  it("ищет без учёта регистра", () => {
    expect(filterDisciplines(discs, analysis, { search: "алгОр" })).toEqual([{ name: "Алгоритмы" }]);
  });

  it("фильтрует по году из анализа", () => {
    expect(filterDisciplines(discs, analysis, { year: 2024 })).toEqual([
      { name: "Базы данных" },
      { name: "Физкультура" },
    ]);
  });

  it("фильтр «с пробелами» — только с gaps > 0", () => {
    expect(filterDisciplines(discs, analysis, { filter: "gaps" })).toEqual([{ name: "Алгоритмы" }]);
  });

  it("фильтр «слабый охват» — ниже 0.5, но не без данных", () => {
    expect(filterDisciplines(discs, analysis, { filter: "low" })).toEqual([{ name: "Базы данных" }]);
  });

  it("фильтр «нет данных» — no_data или нет анализа", () => {
    expect(filterDisciplines(discs, analysis, { filter: "nodata" })).toEqual([{ name: "Физкультура" }]);
    expect(filterDisciplines([{ name: "Новичок" }], analysis, { filter: "nodata" })).toEqual([{ name: "Новичок" }]);
  });

  it("комбинирует поиск + год + фильтр", () => {
    expect(filterDisciplines(discs, analysis, { search: "базы", year: 2024, filter: "low" })).toEqual([
      { name: "Базы данных" },
    ]);
  });
});

describe("disciplineFilterCounts", () => {
  it("считает все четыре категории по одной выборке", () => {
    expect(disciplineFilterCounts(discs, analysis, {})).toEqual({ all: 3, gaps: 1, low: 1, nodata: 1 });
  });

  it("учитывает поиск", () => {
    expect(disciplineFilterCounts(discs, analysis, { search: "алг" })).toEqual({ all: 1, gaps: 1, low: 0, nodata: 0 });
  });
});

describe("sortDisciplines", () => {
  it("по названию — локаль ru, исходный порядок не важен", () => {
    const sorted = sortDisciplines(discs, analysis, "name");
    expect(sorted.map((d) => d.name)).toEqual(["Алгоритмы", "Базы данных", "Физкультура"]);
  });

  it("по охвату — худшие сверху, «нет данных» в конце", () => {
    const sorted = sortDisciplines(discs, analysis, "coverage");
    expect(sorted.map((d) => d.name)).toEqual(["Базы данных", "Алгоритмы", "Физкультура"]);
  });

  it("по пробелам — по убыванию", () => {
    const sorted = sortDisciplines(discs, analysis, "gaps");
    expect(sorted[0].name).toBe("Алгоритмы");
  });

  it("plan — как есть (тот же массив элементов)", () => {
    expect(sortDisciplines(discs, analysis, "plan")).toEqual(discs);
  });
});

const comps = [
  { id: "a", code: "УК-1", name: "Общекультурная", ksa: { knowledge: [{}], abilities: [], skills: [{}] } },
  { id: "b", code: "ОПК-2", name: "Проектная", skills: [] },
  { id: "c", code: "ПК-3.1" },
];

describe("filterCompetencies", () => {
  it("ищет по коду и названию", () => {
    expect(filterCompetencies(comps, { search: "опк" })).toEqual([comps[1]]);
    expect(filterCompetencies(comps, { search: "проектн" })).toEqual([comps[1]]);
  });

  it("фильтрует без ЗУН / с ЗУН", () => {
    expect(filterCompetencies(comps, { filter: "no-ksa" })).toEqual([comps[1], comps[2]]);
    expect(filterCompetencies(comps, { filter: "with-ksa" })).toEqual([comps[0]]);
  });
});

describe("competencyFilterCounts", () => {
  it("делит список на с ЗУН и без", () => {
    expect(competencyFilterCounts(comps)).toEqual({ all: 3, "no-ksa": 2, "with-ksa": 1 });
  });
});

const recs = [
  { discipline_id: "Алгоритмы", competency_id: "УК-1", suggestion_type: "modify" },
  { discipline_id: "Алгоритмы", competency_id: "", suggestion_type: "add" },
  { discipline_id: "Базы данных", competency_id: "УК-1", suggestion_type: "remove" },
];

describe("filterRecommendations", () => {
  it("отбирает по дисциплине", () => {
    expect(filterRecommendations(recs, "Алгоритмы", {})).toHaveLength(2);
  });

  it("дальше — по компетенции и типу", () => {
    expect(filterRecommendations(recs, "Алгоритмы", { comp: "УК-1" })).toHaveLength(1);
    expect(filterRecommendations(recs, "Алгоритмы", { type: "add" })).toHaveLength(1);
    expect(filterRecommendations(recs, "Алгоритмы", { comp: "УК-1", type: "add" })).toHaveLength(0);
  });

  it("пустая дисциплина — весь список", () => {
    expect(filterRecommendations(recs, "", {})).toHaveLength(3);
  });
});
