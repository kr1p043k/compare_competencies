import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge, Button, Skeleton } from "../../design";
import {
  X, MapPin, Building2, Calendar, Star, Tags, FileText,
  ExternalLink,
} from "lucide-react";
import { sanitizeHtml, parseSkillsFromHtml } from "./VacancyCard";
import { reportError } from "./../../lib/errors";
import { request } from "../../lib/api";

interface VacancySummary {
  id: string;
  name: string;
  experience?: string;
  employer_name?: string;
  employer_logo?: string;
  area?: string;
  published_at?: string;
  alternate_url?: string;
}

interface VacancyDetail {
  id: string;
  name?: string;
  description?: string;
  experience?: unknown;
  salary?: unknown;
  employer?: { name?: string; logo_urls?: Record<string, string> };
  employer_logo?: string;
  area?: { name?: string };
  published_at?: string;
  alternate_url?: string;
  skills?: string[];
  key_skills?: unknown[];
  snippet?: unknown;
}

/** Панель деталей вакансии: выезжает справа, вёрстка страницы не меняется. */
export function VacancyDetailPanel({
  vacancy,
  onClose,
}: {
  vacancy: VacancySummary | null;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<VacancyDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!vacancy) return;
    setDetail(null);
    setLoading(true);
    request<VacancyDetail>(`/api/vacancies/${vacancy.id}`)
      .then((d) => setDetail(d))
      .catch((err) => reportError("app/components/VacancyDetailPanel.tsx", err))
      .finally(() => setLoading(false));
  }, [vacancy]);

  const extracted = (detail?.skills ?? []).filter((s: string) => (s || "").trim());
  const parsed = detail?.description ? parseSkillsFromHtml(detail.description) : [];
  const displaySkills = extracted.length > 0 ? extracted : parsed;

  /** «Ключевые навыки» из ответа hh.ru: элементы - строки или { name }. */
  const keySkills = (detail?.key_skills ?? [])
    .map((ks) =>
      typeof ks === "string" ? ks : (ks as { name?: string })?.name ?? "",
    )
    .map((label) => label.trim())
    .filter(Boolean);

  // Логотип: сначала из детального ответа, затем из строки списка.
  const logoUrl =
    detail?.employer_logo ||
    detail?.employer?.logo_urls?.["240"] ||
    detail?.employer?.logo_urls?.["original"] ||
    vacancy?.employer_logo;

  const employerName = detail?.employer?.name || vacancy?.employer_name;

  return (
    <AnimatePresence>
      {vacancy && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-50 flex justify-end bg-overlay"
          onClick={onClose}
        >
          <motion.div
            initial={{ x: 32 }}
            animate={{ x: 0 }}
            exit={{ x: 32 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="flex h-full w-full max-w-2xl flex-col overflow-hidden bg-card shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex shrink-0 items-start justify-between gap-3 border-b border-border px-5 pt-5 pb-3">
              <div className="flex min-w-0 items-center gap-3">
                <div className="grid size-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-border bg-surface">
                  {logoUrl ? (
                    <img
                      src={logoUrl}
                      alt={employerName || ""}
                      className="size-full object-contain p-1.5"
                    />
                  ) : (
                    <Building2 className="size-5 text-muted-foreground" />
                  )}
                </div>
                <div className="min-w-0">
                  <h4 className="text-lg font-semibold leading-snug text-foreground">
                    {detail?.name || vacancy.name}
                  </h4>
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    {employerName && (
                      <span className="inline-flex items-center gap-1">
                        <Building2 className="size-3.5" />
                        {employerName}
                      </span>
                    )}
                    {detail?.area?.name && (
                      <span className="inline-flex items-center gap-1">
                        <MapPin className="size-3.5" />
                        {detail.area.name}
                      </span>
                    )}
                    {detail?.published_at && (
                      <span className="inline-flex items-center gap-1">
                        <Calendar className="size-3.5" />
                        {new Date(detail.published_at).toLocaleDateString("ru-RU", {
                          day: "numeric",
                          month: "long",
                          year: "numeric",
                        })}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="size-9 shrink-0 p-0"
                onClick={onClose}
                title="Скрыть подробности"
              >
                <X className="size-4" />
              </Button>
            </div>

            <div className="flex min-h-0 flex-1 flex-col p-5">
              {loading ? (
                /* Скелет вместо спиннера: повторяет форму контента
                   (бейджи навыков + блок описания) и не прыгает
                   при загрузке данных. */
                <div role="status" className="flex flex-1 flex-col gap-5">
                  <span className="sr-only">Загрузка данных вакансии…</span>
                  <div className="space-y-2.5">
                    <Skeleton className="h-4 w-44" />
                    <div className="flex flex-wrap gap-1.5">
                      <Skeleton className="h-5 w-20 rounded-full" />
                      <Skeleton className="h-5 w-16 rounded-full" />
                      <Skeleton className="h-5 w-24 rounded-full" />
                    </div>
                  </div>
                  <div className="flex flex-1 flex-col gap-2">
                    <Skeleton className="h-4 w-36" />
                    <Skeleton className="min-h-40 w-full rounded-lg" />
                  </div>
                </div>
              ) : (
                <>
                  <div className="shrink-0 space-y-4">
                    {/* Ключевые навыки: работодатель сам выделил их на hh.ru.
                        Акцентный тон primary отличает их от найденных парсером. */}
                    {keySkills.length > 0 && (
                      <section aria-labelledby="vdp-key-skills" className="space-y-2.5">
                        <div className="flex items-center gap-2">
                          <span
                            aria-hidden="true"
                            className="grid size-6 shrink-0 place-items-center rounded-md bg-primary-soft text-primary-soft-foreground"
                          >
                            <Star className="size-3.5" />
                          </span>
                          <h3
                            id="vdp-key-skills"
                            className="text-xs font-semibold uppercase tracking-wider text-foreground"
                          >
                            Ключевые навыки (HH)
                          </h3>
                          <Badge tone="neutral" className="tabular-nums">
                            {keySkills.length}
                          </Badge>
                        </div>
                        <ul className="flex flex-wrap gap-1.5">
                          {keySkills.map((label, index) => (
                            <li key={`${label}-${index}`}>
                              <Badge tone="primary">{label}</Badge>
                            </li>
                          ))}
                        </ul>
                      </section>
                    )}

                    {/* Найденные навыки: результат парсера (или технологии из
                        описания). Нейтральный тон + точка и полужирный текст
                        отделяют их от ключевых навыков работодателя. */}
                    {displaySkills.length > 0 && (
                      <section aria-labelledby="vdp-found-skills" className="space-y-2.5">
                        <div className="flex items-center gap-2">
                          <span
                            aria-hidden="true"
                            className="grid size-6 shrink-0 place-items-center rounded-md bg-info-soft text-info"
                          >
                            <Tags className="size-3.5" />
                          </span>
                          <h3
                            id="vdp-found-skills"
                            className="text-xs font-semibold uppercase tracking-wider text-foreground"
                          >
                            {extracted.length > 0 ? "Найденные навыки" : "Технологии из описания"}
                          </h3>
                          <Badge tone="neutral" className="tabular-nums">
                            {displaySkills.length}
                          </Badge>
                        </div>
                        <ul className="flex flex-wrap gap-1.5">
                          {displaySkills.map((skill: string, index: number) => (
                            <li key={`${skill}-${index}`}>
                              <Badge tone="neutral" className="font-semibold">
                                <span
                                  aria-hidden="true"
                                  className="size-1.5 shrink-0 rounded-full bg-current opacity-70"
                                />
                                {skill}
                              </Badge>
                            </li>
                          ))}
                        </ul>
                      </section>
                    )}

                    {detail?.alternate_url && (
                      <Button asChild className="w-full">
                        <a href={detail.alternate_url} target="_blank" rel="noopener noreferrer">
                          Открыть на hh.ru
                          <ExternalLink className="ml-2 size-4" />
                        </a>
                      </Button>
                    )}
                  </div>

                  <section className="mt-5 flex min-h-0 flex-1 flex-col" aria-labelledby="vdp-description">
                    <div className="mb-2 flex shrink-0 items-center gap-2">
                      <span
                        aria-hidden="true"
                        className="grid size-6 shrink-0 place-items-center rounded-md bg-muted text-foreground"
                      >
                        <FileText className="size-3.5" />
                      </span>
                      <h3
                        id="vdp-description"
                        className="text-xs font-semibold uppercase tracking-wider text-foreground"
                      >
                        Описание вакансии
                      </h3>
                    </div>
                    {detail?.description ? (
                      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
                        <p className="whitespace-pre-line text-justify text-sm leading-relaxed text-foreground/80">
                          {sanitizeHtml(detail.description)}
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">Нет описания</p>
                    )}
                  </section>
                </>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
