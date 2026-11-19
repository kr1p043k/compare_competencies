/**
 * Нечёткий поиск для UI: нормализация, скоринг и фильтрация списков.
 */

const CYR_TO_LAT: Record<string, string> = {
  а: "a",
  б: "b",
  в: "v",
  г: "g",
  д: "d",
  е: "e",
  ж: "zh",
  з: "z",
  и: "i",
  й: "y",
  к: "k",
  л: "l",
  м: "m",
  н: "n",
  о: "o",
  п: "p",
  р: "r",
  с: "s",
  т: "t",
  у: "u",
  ф: "f",
  х: "h",
  ц: "ts",
  ч: "ch",
  ш: "sh",
  щ: "sch",
  ъ: "",
  ы: "y",
  ь: "",
  э: "e",
  ю: "yu",
  я: "ya",
};

function transliterateCyrillic(s: string): string {
  let out = "";
  for (const ch of s) {
    out += CYR_TO_LAT[ch] ?? ch;
  }
  return out;
}

function isWordBoundary(text: string, index: number): boolean {
  if (index <= 0) return true;
  const prev = text[index - 1];
  return /[\s\p{P}\p{S}_]/u.test(prev);
}

const CYRILLIC_RE = /[\u0400-\u04FF]/;

/** Есть ли в строке кириллица (до нормализации, регистр не важен). */
function hasCyrillic(s: string): boolean {
  return CYRILLIC_RE.test(s);
}

export function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/ё/g, "е")
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Привести запрос и текст к одному алфавиту, но ТОЛЬКО когда они разного письма:
 * так работает поиск при неверной раскладке ("птн" на английской раскладке ->
 * "python"), но русский запрос не начинает ловить случайные английские слова.
 */
function alignScripts(query: string, text: string): [string, string] {
  const qCyr = hasCyrillic(query);
  const tCyr = hasCyrillic(text);
  if (qCyr === tCyr) return [query, text];
  return [transliterateCyrillic(query), transliterateCyrillic(text)];
}

function subsequenceScore(q: string, t: string): number {
  if (!q) return 0;
  let qi = 0;
  let first = -1;
  let last = -1;
  let prev = -1;
  let boundaryBonus = 0;
  let consecutiveBonus = 0;

  for (let i = 0; i < t.length && qi < q.length; i++) {
    if (t[i] !== q[qi]) continue;
    if (first === -1) first = i;
    if (isWordBoundary(t, i)) boundaryBonus += 15;
    if (prev !== -1 && i === prev + 1) consecutiveBonus += 10;
    prev = i;
    last = i;
    qi++;
  }

  if (qi !== q.length) return 0;

  const span = last - first + 1;
  const gaps = span - q.length;
  let score = 180 + boundaryBonus + consecutiveBonus - gaps * 3 - first;

  if (score < 1) score = 1;
  if (score > 399) score = 399;
  return score;
}

export function fuzzyScore(query: string, text: string): number {
  const [aq, at] = alignScripts(query, text);
  const q = normalize(aq);
  const t = normalize(at);

  if (!q) return 1;
  if (!t) return 0;

  if (q === t) return 1000;

  if (t.startsWith(q)) {
    return 800 + (q.length / Math.max(t.length, 1)) * 50;
  }

  const idx = t.indexOf(q);
  if (idx !== -1) {
    let score = 600 + (q.length / Math.max(t.length, 1)) * 40 - Math.min(idx, 20);
    if (isWordBoundary(t, idx)) score += 20;
    return score;
  }

  return subsequenceScore(q, t);
}

export function fuzzyMatch(query: string, text: string): boolean {
  return fuzzyScore(query, text) > 0;
}

export function fuzzyFilter<T>(
  items: readonly T[],
  query: string,
  getText: (item: T) => string,
): T[] {
  const q = normalize(query);
  if (!q) return items.slice();

  return items
    .map((item, index) => ({ item, index, score: fuzzyScore(q, getText(item)) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((entry) => entry.item);
}
