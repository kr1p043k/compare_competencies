import { saveBlob } from "./api";

/**
 * Минимальный CSV-helper для экспорта таблиц в Excel.
 *
 * Разделитель - запятая, строки разделяются CRLF, в начале обязателен BOM:
 * без него Excel на Windows открывает кириллицу в ANSI и показывает «крокозябры».
 * Кавычка внутри значения удваивается, само значение берётся в кавычки,
 * если содержит кавычку, перевод строки или разделитель (запятая/точка с запятой).
 */

/** Экранировать одно значение ячейки по правилам RFC 4180. */
export function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",;\r\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

/** Собрать CSV из строк: заголовок по labels, данные в порядке columns. */
export function toCsv(
  rows: Record<string, unknown>[],
  columns: { key: string; label: string }[],
): string {
  const lines = [columns.map((column) => csvCell(column.label)).join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvCell(row[column.key])).join(","));
  }
  return `\uFEFF${lines.join("\r\n")}`;
}

/** Отдать готовый CSV браузеру как файл. */
export function downloadCsv(filename: string, csv: string): void {
  saveBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), filename);
}
