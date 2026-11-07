/**
 * Пояснение метрик преподавательского отчёта.
 *
 * На экране одновременно живут «охват рынка», «покрытие», «качество»,
 * «сильные совпадения», «источник рынка» - без расшифровки они читаются как
 * одно и то же число, хотя считаются по-разному. Здесь у каждой метрики
 * короткое объяснение и подсказка, что она значит на практике.
 */
import { useState } from "react";
import { HelpCircle, ChevronDown, ChevronUp } from "lucide-react";

type Metric = {
  key: string;
  label: string;
  short: string;
  detail: string;
};

export const METRICS: Metric[] = [
  {
    key: "market_top_coverage",
    label: "Охват рынка",
    short: "доля релевантных рыночных навыков, которые реально есть в дисциплине",
    detail:
      "Берётся топ рыночных навыков, отобранных именно для этой дисциплины, и " +
      "считается, сколько из них встречается в её содержании. Это основная " +
      "метрика: именно она показывает, насколько программа отвечает спросу.",
  },
  {
    key: "market_weight_coverage",
    label: "Взвешенный охват",
    short: "то же, но каждый навык весит по своей востребованности",
    detail:
      "Навык с тысячей вакансий весит больше, чем навык с десятью. Показывает, " +
      "закрыт ли массовый спрос, а не просто «сколько галочек».",
  },
  {
    key: "coverage_ratio",
    label: "Покрытие",
    short: "доля навыков дисциплины, найденных на рынке",
    detail:
      "Считается от содержания дисциплины к рынку, а не наоборот. Часто близко " +
      "к 100%, потому что навыки в плане уже приведены к словарю рынка. Поэтому " +
      "для оценки спроса используется «охват рынка», а не эта метрика.",
  },
  {
    key: "quality_coverage",
    label: "Качество совпадений",
    short: "покрытие с учётом надёжности сопоставления",
    detail:
      "Совпадение «Python» и «python-разработка» весит меньше, чем точное. " +
      "Метрика ниже покрытия, если много совпадений приблизительные.",
  },
  {
    key: "strong_coverage",
    label: "Сильные совпадения",
    short: "только точные и близкие совпадения",
    detail:
      "Самый строгий вариант: учитываются лишь совпадения с высокой " +
      "уверенностью. Нижняя оценка - «хуже этого точно не хуже».",
  },
  {
    key: "gaps",
    label: "Разрывы",
    short: "сколько навыков дисциплины не нашлось на рынке",
    detail:
      "Навык есть в программе, но не встречается в вакансиях - либо устарел, " +
      "либо сформулирован иначе. Требует ручной проверки, а не удаления.",
  },
  {
    key: "emerging_growth",
    label: "Растущие",
    short: "сколько растущих рыночных навыков дисциплине не хватает",
    detail:
      "Не «часто встречается», а «растёт по срезам рынка»: навык набирает долю " +
      "вакансий между периодами и при этом отсутствует в дисциплине.",
  },
  {
    key: "market_top_source",
    label: "Источник рынка",
    short: "откуда взялся список рыночных навыков для дисциплины",
    detail:
      "explicit - задан вручную; derived - получен из соседних дисциплин " +
      "направления; relevance - подобран по смыслу. Если источник derived или " +
      "relevance, метрика приблизительная: знаменатель вычислен, а не задан.",
  },
  {
    key: "market_top_reliable",
    label: "Достоверность",
    short: "хватает ли навыков в знаменателе, чтобы верить проценту",
    detail:
      "Если релевантных навыков меньше пяти, процент - шум: 0% при двух " +
      "навыках ничего не говорит о дисциплине.",
  },
];

export function MetricsGlossary({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <div className={compact ? "text-xs" : "text-sm"}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="metrics-glossary"
        className="inline-flex items-center gap-1.5 cursor-pointer text-violet-700 dark:text-violet-300 hover:underline"
      >
        <HelpCircle className="size-4" />
        Как читать метрики
        {open ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
      </button>

      {open && (
        <div
          id="metrics-glossary"
          className="mt-2 rounded-md border border-gray-200 dark:border-slate-700 p-3 space-y-2"
        >
          {METRICS.map((m) => (
            <div key={m.key}>
              <div className="font-medium text-gray-900 dark:text-slate-100">{m.label}</div>
              <div className="text-gray-600 dark:text-slate-400">{m.short}</div>
              <div className="text-gray-500 dark:text-slate-500 mt-0.5">{m.detail}</div>
            </div>
          ))}
          <div className="pt-1 border-t border-gray-200 dark:border-slate-700 text-gray-500 dark:text-slate-500">
            Дисциплины без навыков помечаются «нет данных» и не участвуют в средних:
            0% покрытия при нулевых разрывах - это отсутствие данных, а не результат.
          </div>
        </div>
      )}
    </div>
  );
}

export default MetricsGlossary;
