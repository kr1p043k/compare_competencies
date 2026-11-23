/**
 * Сводные блоки направления: рекомендации, востребованные навыки рынка,
 * разрывы между планом и рынком, тренды навыков.
 *
 * Перенесено из TeacherDashboard.tsx без изменений разметки и классов.
 * Блоки показываются только пока дисциплина не выбрана.
 */
import type { DirectionAnalysis } from "./types";
import { plural } from "./helpers";

// Класс-эквивалент прежнего inline-стиля `card` (radius 8px = rounded-lg,
// padding 14px = p-3.5, margin 10px = mb-2.5).
const card = "rounded-lg border border-border bg-card p-3.5 mb-2.5 shadow-xs";

interface DirectionInsightsProps {
  analysis: DirectionAnalysis;
}

export function DirectionInsights({ analysis }: DirectionInsightsProps) {
  return (
    <>
      {(analysis.recommendations || []).length > 0 && (
        <div className={card}>
          <div className="text-xs font-semibold text-primary mb-2">
            Что улучшить
          </div>
          {(analysis.recommendations || []).map((r, i) => (
            <div
              key={i}
              className={`mb-1.5 rounded-md border px-2.5 py-2 text-xs ${
                r.priority === "high"
                  ? "border-destructive/30 bg-destructive/5"
                  : r.priority === "medium"
                    ? "border-warning/40 bg-warning-soft"
                    : "border-info/30 bg-info-soft"
              }`}
            >
              <div className="flex gap-1.5 items-center mb-1">
                <span
                  className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                    r.priority === "high"
                      ? "bg-destructive/10 text-destructive"
                      : r.priority === "medium"
                        ? "bg-warning-soft text-warning"
                        : "bg-info-soft text-info"
                  }`}
                >
                  {r.priority}
                </span>
                <span className="text-xs text-muted-foreground">{r.type}</span>
              </div>
              <div className="text-foreground leading-snug">{r.message}</div>
            </div>
          ))}
        </div>
      )}

       <details className={card}>
         <summary className="text-xs font-semibold text-primary cursor-pointer select-none">
           Не хватает в плане
         </summary>
         <div className="text-xs text-muted-foreground mb-1.5">
           Навыки из свежих вакансий, которых нет в плане. Происхождение: частота в вакансиях
           {(analysis.top_emerging_across_all || []).some((s) => s.change_pct != null)
             ? " и динамика спроса по срезам рынка"
             : ""}
           .
         </div>
         <div className="flex gap-1 flex-wrap">
            {((analysis.top_emerging_across_all) || []).map((s, i: number) => (
             <span
               key={i}
               className="inline-block px-2 py-0.5 rounded text-xs bg-primary-soft text-primary-soft-foreground m-0.5"
               title={`Вакансий с этим навыком: ${s.frequency}${s.basis ? ` · основа отбора: ${s.basis}` : ""}`}
             >
               {s.skill}
               <span className="ml-1 opacity-70">{s.frequency} вк.</span>
               {s.change_pct != null && (
                 <span
                   className="ml-1 text-success"
                   title={s.basis ? `Основа отбора: ${s.basis} (динамика по срезам рынка)` : "Динамика спроса"}
                 >
                   {s.change_pct > 0 ? "+" : ""}{s.change_pct.toFixed(0)}%
                 </span>
               )}
             </span>
           ))}
         </div>
       </details>

       <details className={card}>
         <summary className="text-xs font-semibold text-primary cursor-pointer select-none">
           Разрывы между планом и рынком
         </summary>
         <div className="text-xs text-muted-foreground mb-1.5">
           Навыки из планов дисциплин (РПД), на которые нет спроса в вакансиях. Число - в скольких
           дисциплинах навык встречается. Происхождение: сравнение РПД с топом рынка при
           teacher-анализе.
         </div>
         <div className="flex gap-1 flex-wrap">
           {(analysis.top_cross_discipline_gaps || []).map((g, i) => (
             <span
               key={i}
               className="inline-block px-2 py-0.5 rounded text-xs bg-destructive/10 text-destructive m-0.5"
               title={`Навык есть в планах ${g.disciplines === 1 ? "одной дисциплине" : `${g.disciplines} дисциплинах`}, но в свежих вакансиях спроса нет`}
             >
               {g.skill}
               <span className="ml-1 opacity-70">
                 {g.disciplines} {plural(g.disciplines, "дисциплина", "дисциплины", "дисциплин")}
               </span>
             </span>
           ))}
         </div>
       </details>

      {analysis.trends && (
        <details className={card}>
          <summary className="text-xs font-semibold text-primary cursor-pointer select-none mb-2">
            Растущие и падающие навыки
          </summary>
        <div className="flex gap-3 flex-wrap">
          {analysis.trends.rising?.length > 0 && (
            <div className={`${card} flex-1 min-w-[200px]`}>
              <div className="text-xs font-semibold text-success mb-2">Растущие навыки</div>
              {analysis.trends.rising.map((t, i) => (
                <div key={i} className="text-xs py-0.5 text-muted-foreground">
                  {t.skill} <span className="text-success">{t.change_pct > 0 ? "+" : ""}{t.change_pct}%</span>
                </div>
              ))}
            </div>
          )}
          {analysis.trends.declining?.length > 0 && (
            <div className={`${card} flex-1 min-w-[200px]`}>
              <div className="text-xs font-semibold text-destructive mb-2">Падающие навыки</div>
              {analysis.trends.declining.map((t, i) => (
                <div key={i} className="text-xs py-0.5 text-muted-foreground">
                  {t.skill} <span className="text-destructive">{t.change_pct}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
        </details>
      )}
    </>
  );
}
