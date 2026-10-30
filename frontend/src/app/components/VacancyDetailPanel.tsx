import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  X, MapPin, Building2, Calendar, Star, Tags, FileText,
  ExternalLink, Loader2, Briefcase,
} from "lucide-react";
import { sanitizeHtml, parseSkillsFromHtml } from "./VacancyCard";
import { reportError } from "./../../lib/errors";
import { request } from "../../lib/api";

interface VacancySummary {
  id: string;
  name: string;
  experience?: string;
  employer_name?: string;
  area?: string;
  published_at?: string;
  alternate_url?: string;
}

interface VacancyDetail {
  id: string;
  name?: string;
  description?: string;
  experience?: any;
  salary?: any;
  employer?: any;
  area?: any;
  published_at?: string;
  alternate_url?: string;
  skills?: string[];
  key_skills?: any[];
  snippet?: any;
}

/** Full-width детали вакансии (как карточка фильтров): сетка не трогается. */
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
            className="h-full w-full max-w-2xl overflow-y-auto bg-white shadow-xl dark:bg-slate-900"
            onClick={(e) => e.stopPropagation()}
          >
          <Card className="border-0 bg-white dark:bg-slate-900 overflow-hidden rounded-none">
            <CardHeader className="border-b border-slate-200 dark:border-slate-800">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-slate-100 dark:bg-slate-800">
                    <Briefcase className="size-5 text-slate-500 dark:text-slate-400" />
                  </div>
                  <div className="min-w-0">
                    <CardTitle className="text-lg leading-snug">
                      {detail?.name || vacancy.name}
                    </CardTitle>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                      {detail?.employer?.name && (
                        <span className="inline-flex items-center gap-1">
                          <Building2 className="size-3.5" />
                          {detail.employer.name}
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
                          {new Date(detail.published_at).toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" })}
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
            <CardContent className="p-5">
              {loading ? (
                <div className="flex items-center justify-center py-10">
                  <Loader2 className="size-6 animate-spin text-slate-400" />
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="space-y-2 min-w-0">
                    <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      <FileText className="size-3" />
                      Описание вакансии
                    </div>
                    {detail?.description ? (
                      <div className="max-h-[420px] overflow-y-auto pr-1">
                        <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-line">
                          {sanitizeHtml(detail.description)}
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-slate-400">Нет описания</p>
                    )}
                  </div>
                  <div className="space-y-4 min-w-0">
                    {detail?.key_skills && detail.key_skills.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                          <Star className="size-3" />
                          Ключевые навыки (HH)
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {(detail.key_skills as any[])
                            .filter((ks: any) => ((typeof ks === "string" ? ks : ks?.name) || "").trim())
                            .map((ks: any) => (
                              <Badge
                                key={typeof ks === "string" ? ks : ks.name}
                                variant="secondary"
                                className="bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
                              >
                                {typeof ks === "string" ? ks : ks.name}
                              </Badge>
                            ))}
                        </div>
                      </div>
                    )}
                    {displaySkills.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
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
                          <ExternalLink className="size-4 ml-2" />
                        </a>
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
