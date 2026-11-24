/**
 * Левая колонка со списком дисциплин: поиск, поповер «Фильтры» и строки.
 *
 * Раньше над списком лежали ещё и выбор набора, scope-бейджи, «Выбрать все /
 * Снять все», блок РПД и KPI. Теперь здесь только то, что относится к выбору
 * дисциплины: поиск, компактные фильтры и сами строки.
 */
import { ListFilter } from "lucide-react";
import { MetricPair } from "../../../design";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import type { DiscFilter, DiscSort } from "../teacherFilters";
import { covTextClass, plural } from "./helpers";
import type { Discipline, DisciplineAnalysisRow, DisciplineDetail } from "./types";

interface DisciplineListProps {
  filtered: Discipline[];
  disciplines: Discipline[];
  analysisByName: Map<string, DisciplineAnalysisRow>;
  selected: DisciplineDetail | null;
  search: string;
  setSearch: (value: string) => void;
  yearFilter: 0 | 2024 | 2025;
  setYearFilter: (value: 0 | 2024 | 2025) => void;
  discFilter: DiscFilter;
  setDiscFilter: (value: DiscFilter) => void;
  discSort: DiscSort;
  setDiscSort: (value: DiscSort) => void;
  discCounts: Record<DiscFilter, number>;
  discFiltersActive: boolean;
  scopeSaving: string | null;
  loadDiscipline: (name: string) => Promise<void>;
  saveScope: (name: string, included: boolean) => Promise<void>;
}

const CHIP_CLASS = (active: boolean) =>
  "text-xs px-2 py-0.5 rounded border cursor-pointer " +
  (active
    ? "border-primary/60 bg-primary-soft text-primary-soft-foreground"
    : "border-border-strong bg-card text-muted-foreground");

const FILTERS: [DiscFilter, string, string][] = [
  ["all", "Все", "Без дополнительного фильтра"],
  ["gaps", "С пробелами", "Есть навыки рынка, которых дисциплина не закрывает"],
  ["low", "Слабый охват", "Охват ниже 50% - в списке подсвечен янтарным/красным"],
  ["nodata", "Нет данных", "Нет данных анализа: вне учёта или без извлечённых навыков"],
];

export function DisciplineList({
  filtered,
  disciplines,
  analysisByName,
  selected,
  search,
  setSearch,
  yearFilter,
  setYearFilter,
  discFilter,
  setDiscFilter,
  discSort,
  setDiscSort,
  discCounts,
  discFiltersActive,
  scopeSaving,
  loadDiscipline,
  saveScope,
}: DisciplineListProps) {
  const activeCount =
    (yearFilter !== 0 ? 1 : 0) + (discFilter !== "all" ? 1 : 0) + (discSort !== "plan" ? 1 : 0);
  const reset = () => {
    setSearch("");
    setYearFilter(0);
    setDiscFilter("all");
    setDiscSort("plan");
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-border p-3">
        <input
          placeholder="Поиск дисциплин..."
          aria-label="Поиск дисциплины"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full rounded-md border border-border bg-card px-3 py-2 text-[13px] text-foreground outline-none"
        />
        <div className="mt-2 flex items-center gap-2">
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="inline-flex items-center gap-1.5 rounded-md border border-border-strong bg-card px-2.5 py-1 text-xs text-foreground cursor-pointer hover:bg-muted"
                title="Видимость строк, фильтры по данным анализа и порядок"
              >
                <ListFilter className="size-3.5 text-subtle-foreground" aria-hidden="true" />
                Фильтры
                {activeCount > 0 && (
                  <span className="rounded-full bg-primary px-1.5 text-[10px] font-semibold text-primary-foreground">
                    {activeCount}
                  </span>
                )}
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-80 space-y-3">
              <div>
                <div
                  className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                  title="Видимость строк уже загруженного списка. Данные не перезагружает: план выбирается в шапке."
                >
                  Показать в списке
                </div>
                <div className="flex flex-wrap gap-1.5" role="group" aria-label="Показать в списке: фильтр видимости">
                  {([0, 2024, 2025] as const).map((y) => (
                    <button
                      key={y}
                      onClick={() => setYearFilter(y)}
                      aria-pressed={yearFilter === y}
                      title={
                        y === 0
                          ? "Все строки выбранного набора без фильтра"
                          : `Только дисциплины, которые входят и в учебный план набора ${y}`
                      }
                      className={CHIP_CLASS(yearFilter === y)}
                    >
                      {y === 0 ? "все" : `в плане ${y}`}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  По данным анализа
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {FILTERS.map(([id, label, hint]) => (
                    <button
                      key={id}
                      onClick={() => setDiscFilter(id)}
                      aria-pressed={discFilter === id}
                      title={hint}
                      className={CHIP_CLASS(discFilter === id)}
                    >
                      {label} ({discCounts[id]})
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label
                  className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                  htmlFor="teacher-disc-sort"
                >
                  Сортировка
                </label>
                <select
                  id="teacher-disc-sort"
                  value={discSort}
                  onChange={(e) => setDiscSort(e.target.value as DiscSort)}
                  className="w-full rounded border border-border-strong bg-card px-2 py-1 text-xs text-foreground outline-none"
                >
                  <option value="plan">По учебному плану</option>
                  <option value="name">По алфавиту</option>
                  <option value="coverage">Худший охват сверху</option>
                  <option value="gaps">Больше пробелов сверху</option>
                </select>
              </div>

              {discFiltersActive && (
                <button
                  onClick={reset}
                  className="text-xs text-primary underline border-0 bg-transparent cursor-pointer"
                >
                  Сбросить фильтры
                </button>
              )}
            </PopoverContent>
          </Popover>
          <span className="ml-auto whitespace-nowrap text-[11px] text-subtle-foreground">
            {filtered.length} из {disciplines.length}
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {filtered.length === 0 && (
          <div className="text-center text-xs text-subtle-foreground py-8 px-4">
            Ничего не найдено
            {discFiltersActive && (
              <>
                {" - "}
                <button
                  onClick={reset}
                  className="text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer underline"
                >
                  сбросить фильтры
                </button>
              </>
            )}
          </div>
        )}
        {filtered.map((d) => {
          // Map.get вместо .find по массиву: O(1), а не O(n) на каждую строку.
          const discAnalysis = analysisByName.get(d.name);
          return (
            <div
              key={d.name}
              className={`flex gap-2 items-start py-2.5 pl-2 pr-3 border-b border-border ${
                selected?.name === d.name ? "bg-primary-soft" : "bg-transparent"
              } ${d.in_scope ?? true ? "opacity-100" : "opacity-55"}`}
            >
              <input
                type="checkbox"
                checked={d.in_scope ?? true}
                aria-label={`Учитывать в анализе: ${d.name}`}
                disabled={scopeSaving === d.name}
                title={
                  (d.scope_source === "methodology" ? "Исключена методологией (можно вернуть). " : "") +
                  "Учитывать в анализе"
                }
                onChange={(e) => saveScope(d.name, e.target.checked)}
                onClick={(e) => e.stopPropagation()}
                className="size-[15px] accent-primary cursor-pointer shrink-0 mt-0.5"
              />
              <div
                className="flex-1 min-w-0 cursor-pointer"
                role="button"
                tabIndex={0}
                onClick={() => loadDiscipline(d.name)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") loadDiscipline(d.name);
                }}
              >
                <div className="flex justify-between flex-1 min-w-0">
                  <div className="text-[13px] font-semibold text-primary truncate flex-1 min-w-0">
                    {d.name}
                    <YearsBadge years={discAnalysis?.years} />
                  </div>
                  {discAnalysis && (
                    <MetricPair
                      className="text-xs"
                      primaryValue={`${(discAnalysis.coverage_ratio * 100).toFixed(0)}%`}
                      primaryTitle="Покрытие: доля навыков дисциплины, подтверждённых рынком"
                      primaryClassName={`font-bold ${covTextClass(discAnalysis.coverage_ratio)}`}
                      secondaryValue={
                        discAnalysis.weighted_coverage != null
                          ? `${(discAnalysis.weighted_coverage * 100).toFixed(0)}%`
                          : undefined
                      }
                      secondaryLabel="кач."
                      secondaryTitle="Качество совпадений: покрытие, взвешенное по спросу. Поля нет в payload - не выводится"
                    />
                  )}
                </div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  {d.scope_source === "methodology" && (
                    <span className="mr-1.5 text-xs text-primary-soft-foreground bg-primary-soft rounded px-1 py-0.5">
                      методология
                    </span>
                  )}
                  {d.scope_source === "custom" && (
                    <span className="mr-1.5 text-xs text-warning bg-warning-soft rounded px-1 py-0.5">вручную</span>
                  )}
                  {d.course != null && `${d.course} курс · `}
                  {d.competencies_count} {plural(d.competencies_count, "комп.", "комп.", "комп.")} · {d.skills_count}{" "}
                  {plural(d.skills_count, "навык", "навыка", "навыков")}
                  {d.abilities_count != null &&
                    ` · ${d.abilities_count} ${plural(d.abilities_count, "умение", "умения", "умений")}`}
                  {d.knowledge_count != null &&
                    ` · ${d.knowledge_count} ${plural(d.knowledge_count, "знание", "знания", "знаний")}`}
                  {discAnalysis && ` / ${discAnalysis.gaps} ${plural(discAnalysis.gaps, "пробел", "пробела", "пробелов")}`}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Метка набора у дисциплины: 2025 / 2024 / «2024/2025», если в обоих планах. */
function YearsBadge({ years }: { years?: number[] }) {
  if (!years || years.length === 0) return null;
  const only = years.length === 1 ? years[0] : null;
  return (
    <span
      title={only ? `Только в наборе ${only}` : `В наборах ${years.join(" и ")}`}
      className={
        "ml-1.5 align-middle text-xs font-medium rounded px-1.5 py-0.5 whitespace-nowrap " +
        (only === 2025 ? "bg-success-soft text-success" : "bg-muted text-muted-foreground")
      }
    >
      {only ?? years.join("/")}
    </span>
  );
}
