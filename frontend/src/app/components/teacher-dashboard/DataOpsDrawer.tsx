/**
 * Дровер «Данные и РПД»: всё служебное, что не нужно при обычном чтении
 * анализа - состав анализа (scope), пересчёт и загрузка РПД.
 *
 * Раньше этот блок занимал половину сайдбара и раскрывался над списком
 * дисциплин. Теперь он открывается по кнопке в шапке и не мешает навигации.
 */
import { useRef } from "react";
import { X } from "lucide-react";
import { Badge, Button, Meter, SectionTitle } from "../../../design";
import { useFocusTrap } from "../../../lib/a11y/useFocusTrap";
import { plural, rpdStageLabel } from "./helpers";
import type { ScopeChain } from "./types";
import type { useRpdProcessing } from "./useRpdProcessing";

interface DataOpsDrawerProps {
  open: boolean;
  onClose: () => void;
  dirCode: string;
  scopeChain: ScopeChain | null;
  scopeSaving: string | null;
  scopeMsg: string;
  onScopeBulk: (included: boolean) => void;
  stale: boolean;
  runLoading: boolean;
  runMsg: string;
  onRun: () => void;
  rpd: ReturnType<typeof useRpdProcessing>;
}

export function DataOpsDrawer({
  open,
  onClose,
  dirCode,
  scopeChain,
  scopeSaving,
  scopeMsg,
  onScopeBulk,
  stale,
  runLoading,
  runMsg,
  onRun,
  rpd,
}: DataOpsDrawerProps) {
  const panelRef = useFocusTrap<HTMLDivElement>(open, onClose);
  const fileRef = useRef<HTMLInputElement | null>(null);

  if (!open) return null;

  const running = rpd.status?.status === "running" || rpd.status?.status === "started";
  const finished = rpd.status?.status === "completed" || rpd.status?.status === "failed";

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="presentation">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} aria-hidden="true" />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Данные и РПД"
        tabIndex={-1}
        className="relative flex h-full w-full max-w-[440px] flex-col overflow-y-auto bg-card shadow-xl outline-none"
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="m-0 text-sm font-semibold text-foreground">Данные и РПД</h2>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Закрыть" title="Закрыть">
            <X className="size-4" />
          </Button>
        </div>

        <div className="space-y-5 p-4">
          <section>
            <SectionTitle
              title="Состав анализа"
              hint="Какие дисциплины плана участвуют в пересчёте. Средние считаются только по ним."
            />
            {scopeChain ? (
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="neutral" title="Все дисциплины учебного плана выбранного набора">
                  в наборе {scopeChain.total}
                </Badge>
                <Badge tone="primary" title="Отмечены галочками «Учитывать в анализе»">
                  в учёте {scopeChain.in_scope}
                </Badge>
                <Badge tone="success" title="Реально участвуют в анализе: в учёте и хватает данных">
                  в анализе {scopeChain.analyzed ?? "-"}
                </Badge>
                <Badge tone="warning" title="Исключены скриптом методики; галочку можно вернуть">
                  исключено методикой {scopeChain.excluded_manual}
                </Badge>
                <Badge tone="neutral" title="Не участвуют в анализе: в плане нет компетенций, навыков или их связей">
                  без данных {scopeChain.excluded_no_data}
                </Badge>
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">Нет данных о составе</div>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button variant="secondary" size="sm" onClick={() => onScopeBulk(true)} disabled={scopeSaving !== null}>
                Выбрать все
              </Button>
              <Button variant="secondary" size="sm" onClick={() => onScopeBulk(false)} disabled={scopeSaving !== null}>
                Снять все
              </Button>
              {stale && <Badge tone="warning">данные устарели - обновите анализ</Badge>}
            </div>
            {scopeMsg && <div className="mt-2 text-xs text-warning">{scopeMsg}</div>}
          </section>

          <section className="border-t border-border pt-4">
            <SectionTitle
              title="Пересчёт анализа"
              hint="Считает охваты, пробелы и рекомендации направления по текущему составу."
            />
            <Button variant="primary" size="sm" loading={runLoading} onClick={onRun}>
              {runLoading ? "Анализ запущен..." : "Запустить анализ"}
            </Button>
            {runMsg && <div className="mt-2 text-xs text-warning">{runMsg}</div>}
          </section>

          <section className="border-t border-border pt-4">
            <SectionTitle
              title="Загрузка РПД"
              hint={`Рабочее направление: ${dirCode}. Загрузка запускает разбор и пересчёт.`}
            />

            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0] || null;
                e.target.value = "";
                if (f) rpd.uploadFile(f);
              }}
            />
            <Button
              variant="primary"
              size="sm"
              className="w-full"
              loading={rpd.uploading}
              onClick={() => fileRef.current?.click()}
            >
              {rpd.uploading ? "Обработка..." : "Загрузить PDF и обработать"}
            </Button>

            {rpd.sources.yandex_covered?.includes(dirCode) && (
              <Button
                variant={rpd.collecting ? "danger" : "secondary"}
                size="sm"
                className="mt-2 w-full"
                onClick={rpd.collect}
                disabled={rpd.cooldown > 0}
              >
                {rpd.collecting
                  ? "Отмена"
                  : rpd.cooldown > 0
                    ? `Подождите ${rpd.cooldown} сек`
                    : "Собрать с Yandex Disk"}
              </Button>
            )}

            <div className="mt-2 flex flex-col gap-1.5 border-t border-dashed border-border pt-2">
              <input
                type="text"
                value={rpd.yandexUrl}
                onChange={(e) => rpd.setYandexUrl(e.target.value)}
                placeholder="Ссылка на папку Yandex Disk (https://disk.360.yandex.ru/d/...)"
                className="w-full rounded-md border border-border-strong bg-card px-2.5 py-2 text-xs text-foreground outline-none"
              />
              <Button
                variant="success"
                size="sm"
                className="w-full"
                onClick={rpd.collect}
                disabled={rpd.collecting || rpd.cooldown > 0 || !rpd.yandexUrl.trim()}
              >
                {rpd.collecting ? "Сбор..." : "Загрузить по ссылке"}
              </Button>
            </div>

            {finished && (
              <div className={`mt-2 text-xs ${rpd.status?.status === "completed" ? "text-success" : "text-destructive"}`}>
                {rpd.msg}
                {rpd.status?.stats?.disciplines != null && rpd.status?.status === "completed" && (
                  <span>
                    {" "}
                    - {rpd.status.stats.disciplines}{" "}
                    {plural(rpd.status.stats.disciplines, "дисциплина", "дисциплины", "дисциплин")}
                  </span>
                )}
              </div>
            )}
            {rpd.busy && running && (
              <div className="mt-2.5">
                <Meter
                  value={rpd.status?.stats?.progress ?? 10}
                  trackClassName="bg-muted"
                  fillClassName="bg-primary"
                />
                <div className="mt-1.5 text-xs text-muted-foreground">
                  Этап: {rpdStageLabel(rpd.status?.stats?.stage)} · {rpd.status?.stats?.progress ?? 10}%
                </div>
              </div>
            )}
            {rpd.msg && !finished && !running && <div className="mt-2 text-xs text-muted-foreground">{rpd.msg}</div>}
          </section>
        </div>
      </div>
    </div>
  );
}
