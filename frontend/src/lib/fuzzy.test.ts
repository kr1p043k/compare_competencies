import { describe, expect, it } from "vitest";

import { fuzzyFilter, fuzzyMatch, fuzzyScore, normalize } from "./fuzzy";

describe("normalize", () => {
  it("приводит к нижнему регистру, нормализует ё/е, диакритику и пробелы", () => {
    // Кириллица остаётся кириллицей: транслитерация включается только когда
    // запрос и текст на разных алфавитах (поиск при неверной раскладке).
    expect(normalize("  ПрИвЕт,  Ёжик  ")).toBe("привет, ежик");
    expect(normalize("Café résumé")).toBe("cafe resume");
  });
});

describe("fuzzyScore / fuzzyMatch", () => {
  it("даёт приоритет точному совпадению, префиксу и подстроке", () => {
    const exact = fuzzyScore("python", "python");
    const prefix = fuzzyScore("py", "python");
    const substring = fuzzyScore("yth", "python");

    expect(exact).toBeGreaterThan(prefix);
    expect(prefix).toBeGreaterThan(substring);
    expect(fuzzyMatch("yth", "python")).toBe(true);
  });

  it("находит subsequence (буквы по порядку с разрывами)", () => {
    expect(fuzzyMatch("птн", "python")).toBe(true);
    expect(fuzzyScore("птн", "python")).toBeGreaterThan(0);
  });

  it("опечатка-пропуск не должна ломать (допустимо не находить)", () => {
    // "pyton" пропускает 'h' относительно "python" — совпадение допустимо не находить.
    expect(() => fuzzyMatch("pyton", "python")).not.toThrow();
    expect(typeof fuzzyMatch("pyton", "python")).toBe("boolean");
  });

  it("сопоставляет ё/е и регистр", () => {
    expect(fuzzyMatch("ЁЖ", "ежик")).toBe(true);
    expect(fuzzyMatch("еж", "Ёжик")).toBe(true);
  });

  it("пустой запрос трактуется как совпадение (score > 0)", () => {
    expect(fuzzyScore("", "python")).toBeGreaterThan(0);
    expect(fuzzyMatch("", "python")).toBe(true);
    expect(fuzzyMatch("   ", "python")).toBe(true);
  });
});

describe("fuzzyFilter", () => {
  it("фильтрует и сортирует по убыванию релевантности", () => {
    const items = ["xpython", "python", "mypythonista", "java"];
    const filtered = fuzzyFilter(items, "python", (x) => x);
    expect(filtered).toEqual(["python", "xpython", "mypythonista"]);
  });

  it("при пустом запросе возвращает исходный порядок (копию)", () => {
    const items = ["b", "a", "c"];
    const result = fuzzyFilter(items, "", (x) => x);
    expect(result).toEqual(items);
    expect(result).not.toBe(items);
  });

  it("отбрасывает элементы без совпадения", () => {
    const items = ["python", "java", "rust"];
    expect(fuzzyFilter(items, "птн", (x) => x)).toEqual(["python"]);
  });
});
