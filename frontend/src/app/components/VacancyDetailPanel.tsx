import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  X, MapPin, Building2, Calendar, Star, Tags, FileText,
  ExternalLink, Loader2,
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
          className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 dark:bg-black/50"
          onClick={onClose}
        >
          <motion.div
            initial={{ x: 32 }}
            animate={{ x: 0 }}
            exit={{ x: 32 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="flex h-full w-full max-w-2xl flex-col bg-white shadow-xl dark:bg-slate-900"
            onClick={(e) => e.stopPropagation()}
          >
            <Card className="flex h-full flex-col gap-0 overflow-hidden rounded-none border-0 bg-white dark:bg-slate-900">
              <CardHeader className="shrink-0 border-b border-slate-200 dark:border-slate-800">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="grid size-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
                      {logoUrl ? (
                        <img
                          src={logoUrl}
                          alt={employerName || ""}
                          className="size-full object-contain p-1.5"
                        />
                      ) : (
                        <Building2 className="size-5 text-slate-400 dark:text-slate-500" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <CardTitle className="text-lg leading-snug">
                        {detail?.name || vacancy.name}
                      </CardTitle>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
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
                  <Button variant="ghost" size="icon" onClick={onClose} title="Скрыть подробности">
                    <X className="size-4" />
                  </Button>
                </div>
              </CardHeader>

              <CardContent className="flex min-h-0 flex-1 flex-col p-5">
                {loading ? (
                  <div className="flex flex-1 items-center justify-center">
                    <Loader2 className="size-6 animate-spin text-slate-400" />
                  </div>
                ) : (
                  <>
                    <div className="shrink-0 space-y-4">
                      {detail?.key_skills && detail.key_skills.length > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            <Star className="size-3" />
                            Ключевые навыки (HH)
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {(detail.key_skills as unknown[])
                              .filter((ks) =>
                                ((typeof ks === "string" ? ks : (ks as { name?: string })?.name) || "").trim(),
                              )
                              .map((ks) => {
                                const label =
                                  typeof ks === "string" ? ks : (ks as { name?: string }).name || "";
                                return (
                                  <Badge
                                    key={label}
                                    variant="secondary"
                                    className="bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
                                  >
                                    {label}
                                  </Badge>
                                );
                              })}
                          </div>
                        </div>
                      )}

                      {displaySkills.length > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            <Tags className="size-3" />
                            {extracted.length > 0 ? "Найденные навыки" : "Технологии из описания"}
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {displaySkills.map((skill: string) => (
                              <Badge
                                key={skill}
                                variant="outline"
                                className="border-slate-200 bg-white text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
                              >
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {detail?.alternate_url && (
                        <Button asChild className="w-full bg-slate-900 text-white hover:bg-slate-800">
                          <a href={detail.alternate_url} target="_blank" rel="noopener noreferrer">
                            Открыть на hh.ru
                            <ExternalLink className="ml-2 size-4" />
                          </a>
                        </Button>
                      )}
                    </div>

                    <div className="mt-5 flex min-h-0 flex-1 flex-col">
                      <div className="mb-2 flex shrink-0 items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        <FileText className="size-3" />
                        Описание вакансии
                      </div>
                      {detail?.description ? (
                        <div className="min-h-0 flex-1 overflow-y-auto pr-1">
                          <p className="whitespace-pre-line text-justify text-sm leading-relaxed text-slate-700 dark:text-slate-300">
                            {sanitizeHtml(detail.description)}
                          </p>
                        </div>
                      ) : (
                        <p className="text-sm text-slate-400">Нет описания</p>
                      )}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
