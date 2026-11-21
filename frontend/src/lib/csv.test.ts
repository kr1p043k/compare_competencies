import { beforeEach, describe, expect, it, vi } from "vitest";

import { saveBlob } from "./api";
import { csvCell, downloadCsv, toCsv } from "./csv";

vi.mock("./api", () => ({ saveBlob: vi.fn() }));

beforeEach(() => {
  vi.mocked(saveBlob).mockClear();
});

describe("csvCell", () => {
  it("удваивает кавычки внутри значения", () => {
    expect(csvCell('Скай "А"')).toBe('"Скай ""А"""');
  });

  it("заключает в кавычки переводы строк", () => {
    expect(csvCell("первая\nвторая")).toBe('"первая\nвторая"');
    expect(csvCell("первая\r\nвторая")).toBe('"первая\r\nвторая"');
  });

  it("экранирует разделители: запятую и точку с запятой", () => {
    expect(csvCell("Москва, Россия")).toBe('"Москва, Россия"');
    expect(csvCell("Python; SQL")).toBe('"Python; SQL"');
  });

  it("пустые значения превращаются в пустую ячейку", () => {
    expect(csvCell(null)).toBe("");
    expect(csvCell(undefined)).toBe("");
    expect(csvCell(0)).toBe("0");
    expect(csvCell(false)).toBe("false");
  });
});

describe("toCsv", () => {
  const columns = [
    { key: "a", label: "Первый" },
    { key: "b", label: "Второй" },
  ];

  it("добавляет BOM и разделяет строки через CRLF", () => {
    const csv = toCsv([{ a: 1, b: "x" }], columns);
    expect(csv.split("\r\n")).toEqual(["\uFEFFПервый,Второй", "1,x"]);
    expect(csv.startsWith("\uFEFF")).toBe(true);
  });

  it("соблюдает порядок колонок и игнорирует лишние ключи", () => {
    const csv = toCsv([{ z: "Z", a: "A", extra: "ignored" }], [
      { key: "a", label: "A" },
      { key: "z", label: "Z" },
    ]);
    expect(csv).toBe("\uFEFFA,Z\r\nA,Z");
  });

  it("для пустого списка возвращает только заголовок с BOM", () => {
    expect(toCsv([], columns)).toBe("\uFEFFПервый,Второй");
  });

  it("экранирует значения каждой ячейки", () => {
    const csv = toCsv(
      [{ name: 'Иванов, "Иван";\r\nмл.', skills: "Python; SQL" }],
      [
        { key: "name", label: "Название" },
        { key: "skills", label: "Навыки" },
      ],
    );
    expect(csv).toBe(
      '\uFEFFНазвание,Навыки\r\n"Иванов, ""Иван"";\r\nмл.","Python; SQL"',
    );
  });
});

describe("downloadCsv", () => {
  it("сохраняет CSV как Blob с UTF-8 и нужным именем файла", () => {
    const csv = "\uFEFFA,B\r\n1,2";
    downloadCsv("vacancies-2026-11-21.csv", csv);

    expect(saveBlob).toHaveBeenCalledTimes(1);
    const [blob, filename] = vi.mocked(saveBlob).mock.calls[0];
    expect(filename).toBe("vacancies-2026-11-21.csv");
    expect(blob.type).toBe("text/csv;charset=utf-8");
    expect(blob.size).toBe(new Blob([csv]).size);
  });
});
