/**
 * Чистое распознавание данных chart-блока ассистента.
 *
 * Отделено от recharts-рендера (ChatChartBlock.tsx), чтобы:
 *  - распознавание можно было тестировать без jsdom-графики;
 *  - recharts оставался в ленивом чанке, пока не придёт реальный график.
 *
 * LLM присылает форму, близкую к примерам `_blocks_prompt`
 * (src/services/llm_chat.py), но поля могут называться по-разному. Здесь
 * принимаются устойчивые написания: данные ищем в data/points/values/items,
 * серии - в series, тип диаграммы - в chart_type/chartType/kind/variant.
 * Ничего не выдумываем: не распознали - вернём null, и вызывающий покажет
 * фолбэк.
 */

import type { ChatBlock } from "./ChatBlockView";

export type ChartKind = "bar" | "line" | "area" | "pie";

export interface ChartData {
  kind: ChartKind;
  /** Ключ подписи категории на оси X (или имени сектора). */
  labelKey: string;
  /** Ключи числовых серий. */
  seriesKeys: string[];
  /** Строки данных: значение каждой series - число (или строка как есть). */
  rows: Array<Record<string, string | number>>;
  title: string;
}

const MAX_ROWS = 30;

function strOf(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isNumericLike(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value === "string" && value.trim() !== "") {
    return Number.isFinite(Number(value));
  }
  return false;
}

function firstArray(block: ChatBlock): unknown[] | null {
  for (const key of ["data", "points", "values", "items"]) {
    const v = block[key];
    if (Array.isArray(v) && v.length > 0) return v;
  }
  return null;
}

/** Серии: явный список ключей (строкой или {key}) либо числовый вывод
 * из строк данных. `labelKey` из серий исключается. */
function deduceSeriesKeys(block: ChatBlock, sample: Record<string, unknown>, labelKey: string): string[] {
  const declared = block.series;
  if (Array.isArray(declared) && declared.length > 0) {
    const keys = declared
      .map((s) => {
        if (typeof s === "string") return s;
        if (s && typeof s === "object" && !Array.isArray(s)) {
          const rec = s as Record<string, unknown>;
          const key = strOf(rec.key) || strOf(rec.name) || strOf(rec.field);
          return key && key !== labelKey ? key : "";
        }
        return "";
      })
      .filter(Boolean);
    if (keys.length > 0) return keys;
  }
  const keys = Object.keys(sample).filter((k) => k !== labelKey && isNumericLike(sample[k]));
  return keys;
}

/** Формы строк: {"name": ..., "value": ...} | {"label": ..., "<любой ключ>": n} */
export function extractChartData(block: ChatBlock): ChartData | null {
  const rawRows = firstArray(block);
  if (!rawRows) return null;

  let labelKey = "label";
  let seriesKeys: string[] = [];
  let rows: Array<Record<string, string | number>>;

  if (rawRows.every((r) => typeof r === "number" || typeof r === "string")) {
    // плоский список чисел (или строк-цифр): категории - порядковые номера
    labelKey = "label";
    seriesKeys = ["value"];
    rows = rawRows
      .filter((v) => isNumericLike(v))
      .map((v, i) => ({ label: String(i + 1), value: Number(v) }));
  } else {
    const objects = rawRows.filter(
      (r): r is Record<string, unknown> => !!r && typeof r === "object" && !Array.isArray(r),
    );
    if (objects.length === 0) return null;

    // ключ подписи: name/label/x/category, иначе первый нечисловой ключ
    const sample = objects[0];
    labelKey = ["name", "label", "x", "category", "key"].find((k) => k in sample) ?? "";
    if (!labelKey) {
      labelKey =
        Object.keys(sample).find(
          (k) => typeof sample[k] === "string" && !isNumericLike(sample[k]),
        ) ?? "";
    }
    if (!labelKey) labelKey = "name";

    seriesKeys = deduceSeriesKeys(block, sample, labelKey);
    if (seriesKeys.length === 0) return null;

    rows = objects.slice(0, MAX_ROWS).map((r) => {
      const row: Record<string, string | number> = {};
      const rawLabel = r[labelKey];
      row[labelKey] = strOf(rawLabel) || (typeof rawLabel === "number" ? String(rawLabel) : "—");
      for (const key of seriesKeys) {
        const v = r[key];
        row[key] =
          typeof v === "number"
            ? v
            : typeof v === "string" && Number.isFinite(Number(v))
              ? Number(v)
              : 0;
      }
      return row;
    });
  }

  if (rows.length === 0 || seriesKeys.length === 0) return null;

  // тип диаграммы
  const kindRaw = (
    strOf(block.chart_type) ||
    strOf(block.chartType) ||
    strOf(block.kind) ||
    strOf(block.variant)
  ).toLowerCase();
  const kind: ChartKind =
    kindRaw === "line" || kindRaw === "area" || kindRaw === "pie" ? kindRaw : "bar";

  const title = strOf(block.title) || strOf(block.name) || strOf(block.caption);

  return { kind, labelKey, seriesKeys, rows, title };
}
