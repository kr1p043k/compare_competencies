import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  X, MapPin, Building2, Calendar, Star, Tags, FileText,
  ExternalLink, Loader2,
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

export function VacancyDrawer({
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const extracted = (detail?.skills ?? []).filter((s: string) => (s || "").trim());
  const parsed = detail?.description ? parseSkillsFromHtml(detail.description) : [];
  const displaySkills = extracted.length > 0 ? extracted : parsed;

  return (
    <AnimatePresence>
      {vacancy && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-slate-950/50"
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "tween", duration: 0.22, ease: "easeOut" }}
            className="fixed top-0 right-0 bottom-0 z-50 w-full sm:max-w-xl bg-white dark:bg-slate-950 border-l border-gray-200 dark:border-slate-700 shadow-2xl flex flex-col"
          >
            <div className="flex items-start justify-between gap-3 p-5 border-b border-gray-200 dark:border-slate-700">
              <div className="min-w-0">
                <h3 className="text-xl font-bold text-slate-900 dark:text-white leading-snug">
                  {detail?.name || vacancy.name}
                </h3>
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-slate-600 dark:text-slate-400">
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
                </div>
              </div>
              <button
                onClick={onClose}
                aria-label="Закрыть"
                className="p-2 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-800 cursor-pointer shrink-0"
              >
                <X className="size-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="size-6 animate-spin text-slate-400" />
                </div>
              ) : (
                <>
                  {detail?.description && (
                    <section className="space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                        <FileText className="size-3" />
                        Описание вакансии
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                        <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-line">
                          {sanitizeHtml(detail.description)}
                        </div>
                      </div>
                    </section>
                  )}

                  {detail?.key_skills && detail.key_skills.length > 0 && (
                    <section className="space-y-2">
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
                    </section>
                  )}

                  {displaySkills.length > 0 && (
                    <section className="space-y-2">
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
                    </section>
                  )}

                  {detail?.published_at && (
                    <p className="text-xs text-slate-400 dark:text-slate-500 inline-flex items-center gap-1">
                      <Calendar className="size-3" />
                      {new Date(detail.published_at).toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" })}
                    </p>
                  )}
                </>
              )}
            </div>

            {detail?.alternate_url && (
              <div className="p-4 border-t border-gray-200 dark:border-slate-700">
                <Button asChild className="w-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 text-white">
                  <a href={detail.alternate_url} target="_blank" rel="noopener noreferrer">
                    Открыть на hh.ru
                    <ExternalLink className="size-4 ml-2" />
                  </a>
                </Button>
              </div>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
