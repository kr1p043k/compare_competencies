/**
 * Мелкие helper-функции TeacherDashboard.
 *
 * Перенесено из TeacherDashboard.tsx без изменений поведения.
 */
export function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

export const RPD_STAGE_LABELS: Record<string, string> = {
  saved: "файл сохранён",
  parse: "разбор PDF и извлечение компетенций",
  collect: "скачивание аннотаций с Yandex Disk",
  merge: "слияние аннотаций в KRM",
  seed: "запись направления в базу данных",
  analysis: "пересчёт анализа направления",
  done: "готово",
  error: "ошибка",
};

export function rpdStageLabel(stage?: string): string {
  if (!stage) return "...";
  return RPD_STAGE_LABELS[stage] ?? stage;
}

export function covBorderClass(cov: number) {
  return cov > 0.5 ? "border-success" : cov > 0.2 ? "border-warning" : "border-destructive";
}

export function covTextClass(cov: number) {
  return cov > 0.5 ? "text-success" : cov > 0.2 ? "text-warning" : "text-destructive";
}
