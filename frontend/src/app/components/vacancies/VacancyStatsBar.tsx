import { Download } from "lucide-react";

import { Button } from "../../../design";
import { download, saveBlob } from "../../../lib/api";
import { cn } from "../ui/utils";
import type { AppliedFilters } from "./types";

interface VacancyStatsBarProps {
  total: number;
  visible: number;
  canRunPipeline: boolean;
  applied: AppliedFilters;
}

/** Строка «Найдено N вакансий» + экспорт в Excel по применённым фильтрам. */
export function VacancyStatsBar({ total, visible, canRunPipeline, applied }: VacancyStatsBarProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <p className="text-sm text-muted-foreground">
        Найдено <span className="font-semibold text-foreground dark:text-muted-foreground">{total}</span> вакансий
        {visible < total && (
          <span> (показано {visible})</span>
        )}
      </p>
      <Button
        variant="secondary"
        size="sm"
        title={canRunPipeline === false ? "Экспорт доступен преподавателям" : undefined}
        onClick={async () => {
          try {
            const eq = new URLSearchParams();
            if (applied.search.trim()) eq.append("search", applied.search.trim());
            if (applied.experience !== "all") eq.append("experience", applied.experience);
            if (applied.city !== "all") eq.append("region", applied.city);
            if (applied.months) eq.append("months", String(applied.months));
            if (applied.date_from) eq.append("date_from", applied.date_from);
            if (applied.date_to) eq.append("date_to", applied.date_to);
            const qs = eq.toString();
            const { blob, filename } = await download(
              `/api/teacher/export/vacancies${qs ? `?${qs}` : ""}`,
            );
            saveBlob(
              blob,
              filename ?? `vacancies_skills_${new Date().toISOString().split("T")[0]}.xlsx`,
            );
          } catch (e) {
            console.error("Export failed:", e);
          }
        }}
        className={cn("gap-2", canRunPipeline === false && "hidden")}
      >
        <Download className="size-4" />
        Excel
      </Button>
    </div>
  );
}
