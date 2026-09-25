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
    fetch(`/api/vacancies/${vacancy.id}`)
      .then((r) => r.json())
      .then((d) => setDetail(d))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [vacancy]);

  const extracted = (detail?.skills ?? []).filter((s: string) => (s || "").trim());
  const parsed = detail?.description ? parseSkillsFromHtml(detail.description) : [];
  const displaySkills = extracted.length > 0 ? extracted : parsed;

  return (
    <AnimatePresence>
      {vacancy && (
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden">
            <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="p-2 bg-gradient-to-br from-blue-500 dark:from-blue-950/30 to-purple-600 rounded-lg shadow-md shrink-0">
                    <Briefcase className="size-5 text-white" />
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
                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="space-y-2 min-w-0">
                    <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      <FileText className="size-3" />
                      Описание вакансии
                    </div>
                    {detail?.description ? (
                      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700 max-h-96 overflow-y-auto">
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
                                className="bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800"
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
                              className="bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800"
                            >
                              {skill}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {detail?.alternate_url && (
                      <Button asChild className="w-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 text-white">
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
      )}
    </AnimatePresence>
  );
}
