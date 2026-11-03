import { describe, expect, it } from "vitest";

import { matchesQuery, shortenAbstract, type Publication } from "./publicationsFilter";

const basePub: Publication = {
  id: 1,
  title: "Моделирование графов в обучении",
  authors: ["Егоров Александр Петрович", "Иванова Мария"],
  type_id: 39,
  type_name: "Научная статья",
  signature: "Моделирование графов // Егоров А.П.",
  doi: "10.1234/abc",
  isbn: null,
  track_id: "801000001",
  created_at: "2026-10-01 12:00:00",
  year: 2025,
  academic_subject: ["Математика"],
  education_program: ["01.03.01 Математика"],
  abstract: "Рассматриваются алгоритмы обхода графа и их применение на практике.",
  keywords: ["графы", "алгоритмы"],
  journal: "Вестник ЮФУ",
  publisher: "Издательство ЮФУ",
  language: "ru",
  publishing_year: 2025,
  pages: "12-25",
  files: [],
  url: "https://hub.sfedu.ru/api/assets/1/",
  web_url: "https://hub.sfedu.ru/dashboard/asset/1/",
  creator_username: "egorov",
};

describe("matchesQuery", () => {
  it("возвращает true для пустого запроса", () => {
    expect(matchesQuery(basePub, "")).toBe(true);
    expect(matchesQuery(basePub, "   ")).toBe(true);
  });

  it("ищет по названию без учёта регистра", () => {
    expect(matchesQuery(basePub, "ГРАФОВ")).toBe(true);
    expect(matchesQuery(basePub, "графов")).toBe(true);
  });

  it("ищет по фамилии автора", () => {
    expect(matchesQuery(basePub, "егоров")).toBe(true);
    expect(matchesQuery(basePub, "петрович")).toBe(true);
  });

  it("ищет по журналу, издателю, ключевым словам и аннотации", () => {
    expect(matchesQuery(basePub, "вестник")).toBe(true);
    expect(matchesQuery(basePub, "издательство юфу")).toBe(true);
    expect(matchesQuery(basePub, "алгоритмы")).toBe(true);
    expect(matchesQuery(basePub, "обхода")).toBe(true);
  });

  it("требует все слова запроса", () => {
    expect(matchesQuery(basePub, "графов математика")).toBe(true);
    expect(matchesQuery(basePub, "графов физика")).toBe(false);
  });

  it("не находит отсутствующий текст", () => {
    expect(matchesQuery(basePub, "квантовая хромодинамика")).toBe(false);
  });

  it("устойчив к null-полям", () => {
    const sparse: Publication = {
      ...basePub,
      title: null,
      authors: [],
      abstract: null,
      journal: null,
      publisher: null,
      keywords: [],
      academic_subject: [],
      education_program: [],
    };
    expect(matchesQuery(sparse, "графов")).toBe(false);
    expect(matchesQuery(sparse, "")).toBe(true);
  });
});

describe("shortenAbstract", () => {
  it("не трогает короткий текст", () => {
    expect(shortenAbstract("коротко")).toBe("коротко");
  });

  it("возвращает null для пустого значения", () => {
    expect(shortenAbstract(null)).toBeNull();
    expect(shortenAbstract("")).toBeNull();
  });

  it("обрезает длинный текст по границе слова", () => {
    const text = `${"слово ".repeat(60)}конец`;
    const result = shortenAbstract(text, 50);
    expect(result).not.toBeNull();
    expect(result!.endsWith("...")).toBe(true);
    expect(result!.length).toBeLessThanOrEqual(53);
    expect(result!).not.toContain("сло...");
  });
});
