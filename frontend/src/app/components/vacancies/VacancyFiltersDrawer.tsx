import type { KeyboardEvent } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Search, X, Filter, Loader2 } from "lucide-react";

import { Badge, Button, inputClass } from "../../../design";
import { cn } from "../ui/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { fmtDateRU, PERIOD_PRESETS } from "./constants";
import type { AppliedFilters, VacancyInfo } from "./types";

interface VacancyFiltersDrawerProps {
  open: boolean;
  onClose: () => void;
  activeFilterCount: number;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  onSearchKeyPress: (e: KeyboardEvent<HTMLInputElement>) => void;
  onSearch: () => void;
  loading: boolean;
  experienceFilter: string;
  onExperienceChange: (value: string) => void;
  cityFilter: string;
  onCityChange: (value: string) => void;
  allCityOptions: string[];
  sourceFilter: string;
  onSourceChange: (value: string) => void;
  monthsFilter: number | null;
  onMonthsChange: (value: number | null) => void;
  dateFrom: string;
  onDateFromChange: (value: string) => void;
  dateTo: string;
  onDateToChange: (value: string) => void;
  vacancyInfo: VacancyInfo | null;
  applied: AppliedFilters;
  onApplyFilters: (over?: Partial<AppliedFilters>) => void;
  total: number;
  onClearFilters: () => void;
}

/** Выезжающая справа панель фильтров. Состояние живёт в VacanciesList. */
export function VacancyFiltersDrawer({
  open,
  onClose,
  activeFilterCount,
  searchQuery,
  onSearchQueryChange,
  onSearchKeyPress,
  onSearch,
  loading,
  experienceFilter,
  onExperienceChange,
  cityFilter,
  onCityChange,
  allCityOptions,
  sourceFilter,
  onSourceChange,
  monthsFilter,
  onMonthsChange,
  dateFrom,
  onDateFromChange,
  dateTo,
  onDateToChange,
  vacancyInfo,
  applied,
  onApplyFilters,
  total,
  onClearFilters,
}: VacancyFiltersDrawerProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-overlay"
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "tween", duration: 0.22, ease: "easeOut" }}
            className="fixed top-0 right-0 bottom-0 z-50 w-full sm:max-w-lg bg-background border-l border-border dark:border-border shadow-sm flex flex-col"
          >
            <div className="flex items-center justify-between gap-3 p-5 border-b border-border dark:border-border">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary p-2">
                  <Filter className="size-5 text-primary-foreground" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">Фильтры и поиск</div>
                  <div className="text-xs text-muted-foreground">
                    {activeFilterCount > 0 ? `Активно: ${activeFilterCount}` : "Показаны все вакансии"}
                  </div>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={onClose} title="Закрыть" aria-label="Закрыть фильтры" className="size-9 shrink-0 p-0">
                <X className="size-4" />
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {/* Search */}
            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                Поиск по названию
              </label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
                  <input
                    value={searchQuery}
                    onChange={(e) => onSearchQueryChange(e.target.value)}
                    onKeyPress={onSearchKeyPress}
                    placeholder="Должность или компания"
                    className={cn(inputClass, "pl-10")}
                  />
                </div>
                <Button
                  onClick={onSearch}
                  disabled={loading}
                  aria-label="Найти"
                  className="h-10 w-10 p-0"
                >
                  {loading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Search className="size-4" />
                  )}
                </Button>
              </div>
            </div>

            <div className="border-t border-border dark:border-border" />

            {/* Experience + city */}
            <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                Опыт
              </label>
              <Select value={experienceFilter} onValueChange={(v) => { onExperienceChange(v); }}>
                <SelectTrigger className="h-10 rounded-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Все уровни</SelectItem>
                  <SelectItem value="junior">Junior</SelectItem>
                  <SelectItem value="middle">Middle</SelectItem>
                  <SelectItem value="senior">Senior</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* City filter */}
            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                Город
              </label>
              <Select value={cityFilter} onValueChange={(v) => { onCityChange(v); }}>
                <SelectTrigger className="h-10 rounded-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-80">
                  <SelectItem value="all">Все города</SelectItem>
                  {allCityOptions.map(city => (
                    <SelectItem key={city} value={city}>
                      {city}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            </div>

            <div className="border-t border-border dark:border-border" />

            {/* Source: hh.ru или Работа.ру */}
            <div className="space-y-2">
              <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                Источник
              </label>
              <Select value={sourceFilter} onValueChange={(v) => { onSourceChange(v); }}>
                <SelectTrigger className="h-10 rounded-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Все источники</SelectItem>
                  <SelectItem value="hh">hh.ru</SelectItem>
                  <SelectItem value="rabota">Работа.ру</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="border-t border-border dark:border-border" />

            {/* Period presets */}
            <div className="space-y-3">
              <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                Период
              </label>
              <div className="flex flex-wrap gap-2">
                {PERIOD_PRESETS.map(p => {
                  const isActive = monthsFilter === p.value && !dateFrom && !dateTo;
                  const dimmed = (dateFrom !== "" || dateTo !== "") && p.value !== null;
                  return (
                    <Button
                      key={p.label}
                      type="button"
                      variant={isActive ? "primary" : "secondary"}
                      size="sm"
                      onClick={() => { onMonthsChange(p.value); if (p.value !== null) { onDateFromChange(""); onDateToChange(""); } }}
                      className={cn("h-9 rounded-full px-4 text-sm font-medium", dimmed && "opacity-40")}
                    >
                      {p.label}
                    </Button>
                  );
                })}
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="date"
                  value={dateFrom}
                  min={vacancyInfo?.date_range?.from?.slice(0, 10)}
                  max={dateTo || vacancyInfo?.date_range?.to?.slice(0, 10)}
                  onChange={(e) => { onDateFromChange(e.target.value); if (e.target.value) onMonthsChange(null); }}
                  className={cn(inputClass, "dark:[color-scheme:dark]")}
                  aria-label="Дата от"
                />
                <span className="text-sm text-muted-foreground">-</span>
                <input
                  type="date"
                  value={dateTo}
                  min={dateFrom || vacancyInfo?.date_range?.from?.slice(0, 10)}
                  max={vacancyInfo?.date_range?.to?.slice(0, 10)}
                  onChange={(e) => { onDateToChange(e.target.value); if (e.target.value) onMonthsChange(null); }}
                  className={cn(inputClass, "dark:[color-scheme:dark]")}
                  aria-label="Дата до"
                />
              </div>
              {vacancyInfo?.date_range && (
                <p className="text-xs text-muted-foreground">
                  Записи с {fmtDateRU(vacancyInfo.date_range.from)}
                </p>
              )}
            </div>
          </div>
          </div>

          {/* Active filters */}
          {(applied.experience !== "all" || applied.city !== "all" || (applied.source && applied.source !== "all") || applied.search || applied.months !== null || applied.date_from || applied.date_to) && (
            <div className="flex flex-wrap items-center gap-2 px-6 pt-4 border-t border-border dark:border-border">
              {applied.source && applied.source !== "all" && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => { onSourceChange("all"); onApplyFilters({ source: "all" }); }}
                >
                  {applied.source === "rabota" ? "Работа.ру" : "hh.ru"} ✕
                </Badge>
              )}
              {applied.experience !== "all" && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => { onExperienceChange("all"); onApplyFilters({ experience: "all" }); }}
                >
                  {applied.experience} ✕
                </Badge>
              )}
              {applied.city !== "all" && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => { onCityChange("all"); onApplyFilters({ city: "all" }); }}
                >
                  {applied.city} ✕
                </Badge>
              )}
              {applied.search && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => {
                    onSearchQueryChange("");
                    onApplyFilters({ search: "" });
                  }}
                >
                  "{applied.search}" ✕
                </Badge>
              )}
              {applied.months !== null && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => { onMonthsChange(null); onApplyFilters({ months: null }); }}
                >
                  {applied.months} мес ✕
                </Badge>
              )}
              {(applied.date_from || applied.date_to) && (
                <Badge
                  tone="neutral"
                  className="cursor-pointer hover:bg-foreground/10"
                  onClick={() => { onDateFromChange(""); onDateToChange(""); onApplyFilters({ date_from: "", date_to: "" }); }}
                >
                  {applied.date_from ? fmtDateRU(applied.date_from) : "…"} - {applied.date_to ? fmtDateRU(applied.date_to) : "…"} ✕
                </Badge>
              )}
            </div>
          )}
            <div className="flex items-center gap-3 p-5 border-t border-border dark:border-border bg-background">
              <Button
                onClick={() => { onApplyFilters(); onClose(); }}
                disabled={loading}
                className="gap-2 h-10 flex-1 rounded-lg whitespace-nowrap hover:bg-primary/90 hover:brightness-100"
              >
                {loading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Search className="size-4" />
                )}
                Показать вакансии{total > 0 ? ` (${total})` : ""}
              </Button>
              <Button
                onClick={onClearFilters}
                disabled={loading || activeFilterCount === 0}
                variant="secondary"
                className="gap-2 h-10 rounded-lg whitespace-nowrap text-muted-foreground"
              >
                <X className="size-4" />
                Сбросить
              </Button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
