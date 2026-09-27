import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import {
  Shield,
  Users,
  FileText,
  Activity,
  BookOpen,
  Loader2,
  User,
  Plus,
  X,
  CheckCircle2,
  ChevronDown,
  GraduationCap,
  Database,
  AlertCircle,
} from "lucide-react";
import { apiFetch, useAuth } from "../../lib/auth";

type AccountProps = { displayName?: string; email?: string };

function AccountCard({ displayName, email, roleLabel }: AccountProps & { roleLabel: string }) {
  return (
    <Card className="bg-white dark:bg-slate-950">
      <CardHeader>
        <CardTitle className="text-lg text-slate-900 dark:text-slate-100">Учётная запись</CardTitle>
        <CardDescription className="text-slate-600 dark:text-slate-400">{roleLabel}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-sm text-slate-900 dark:text-slate-100">
        <p>
          <span className="text-slate-600 dark:text-slate-400">ФИО: </span>
          {displayName || "—"}
        </p>
        <p>
          <span className="text-slate-600 dark:text-slate-400">Email: </span>
          {email || "—"}
        </p>
      </CardContent>
    </Card>
  );
}

/** Полная страница профиля администратора: учётка + live-сводка + ссылки на админ-разделы. */
export function AdminProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  const [sum, setSum] = useState<{
    sessions_active?: number | null;
    errors_1h?: { "4xx"?: number; "5xx"?: number };
    freshness?: { vacancies?: number; date_from?: string | null; date_to?: string | null };
  } | null>(null);
  const [sumLoading, setSumLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch("/api/admin/monitoring/summary");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (!cancelled) setSum(d);
      } catch {
        if (!cancelled) setSum(null);
      } finally {
        if (!cancelled) setSumLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const errorsTotal =
    sum && sum.errors_1h
      ? (sum.errors_1h["4xx"] ?? 0) + (sum.errors_1h["5xx"] ?? 0)
      : null;

  return (
    <div className="space-y-6">
      <AccountCard displayName={displayName} email={email} roleLabel="Администратор" />
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2 text-slate-900 dark:text-slate-100">
            <Activity className="size-5 text-blue-600 dark:text-blue-400" />
            Живая сводка
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">
            Активность, ошибки и свежесть данных — прочерк при недоступности
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sumLoading ? (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2" aria-label="Загрузка сводки">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg border border-slate-200 dark:border-slate-700 p-3 animate-pulse">
                  <div className="h-6 w-12 rounded bg-slate-200 dark:bg-slate-700 mx-auto" />
                  <div className="mt-2 h-3 w-20 rounded bg-slate-200 dark:bg-slate-700 mx-auto" />
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-center transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {typeof sum?.sessions_active === "number" ? sum.sessions_active : "—"}
                </div>
                <div className="text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">
                  Сессии активны
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-center transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {errorsTotal === null ? "—" : errorsTotal}
                </div>
                <div className="text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">
                  Ошибки за час
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-3 text-center transition-colors duration-200 col-span-2 lg:col-span-1 dark:border-slate-700 dark:bg-slate-900">
                <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {typeof sum?.freshness?.vacancies === "number" ? sum.freshness.vacancies.toLocaleString("ru-RU") : "—"}
                </div>
                <div className="text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">
                  Вакансии{sum?.freshness?.date_from ? ` · ${sum.freshness.date_from}–${sum.freshness.date_to ?? "…"}` : ""}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2 text-slate-900 dark:text-slate-100">
            <Shield className="size-5 text-blue-600 dark:text-blue-400" />
            Администрирование
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">Быстрые ссылки на административные разделы</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => onNavigate("admin")} className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none">
            <Users className="mr-2 size-4" />
            Пользователи
          </Button>
          <Button variant="outline" onClick={() => onNavigate("logs")} className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none">
            <FileText className="mr-2 size-4" />
            Логи
          </Button>
          <Button variant="outline" onClick={() => onNavigate("monitoring")} className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none">
            <Activity className="mr-2 size-4" />
            Мониторинг
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

type DisciplineRow = {
  name: string;
  competencies_count?: number;
  skills_count?: number;
  semester?: number | null;
  in_scope?: boolean;
};

/** Полная страница профиля преподавателя: учётка + зона преподавания + ссылки на анализ. */
export function TeacherProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate?: (tab: string) => void }) {
  const [discs, setDiscs] = useState<DisciplineRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/teacher/krm/disciplines?dir_code=09.03.02")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => { if (!cancelled) setDiscs(Array.isArray(d) ? d : []); })
      .catch((e: any) => { if (!cancelled) setError(e?.message || "Не удалось загрузить дисциплины"); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      <AccountCard displayName={displayName} email={email} roleLabel="Преподаватель" />
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2 text-slate-900 dark:text-slate-100">
            <BookOpen className="size-5 text-emerald-600 dark:text-emerald-400" />
            Зона преподавания
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">Дисциплины направления 09.03.02 (только просмотр)</CardDescription>
        </CardHeader>
        <CardContent>
          {discs === null && !error && (
            <p className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <Loader2 className="size-4 animate-spin" />
              Загрузка…
            </p>
          )}
          {error && (
            <p className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
              <AlertCircle className="size-4 shrink-0" />
              {error}
            </p>
          )}
          {discs !== null && !error && (
            discs.length === 0 ? (
              <p className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 px-3 py-4 text-center text-sm text-slate-600 dark:text-slate-400">
                Дисциплины не найдены.
              </p>
            ) : (
              <ul className="divide-y divide-gray-100 dark:divide-slate-800">
                {discs.map((d) => (
                  <li key={d.name} className="py-2 flex items-center justify-between gap-3">
                    <span className="text-sm text-slate-900 dark:text-slate-100 min-w-0">
                      <span className="block truncate">{d.name}</span>
                      <span className="flex items-center gap-1.5 mt-1 flex-wrap">
                        {typeof d.semester === "number" && (
                          <Badge variant="outline" className="text-xs">сем. {d.semester}</Badge>
                        )}
                        {typeof d.competencies_count === "number" && (
                          <Badge variant="secondary" className="text-xs">компетенций: {d.competencies_count}</Badge>
                        )}
                        {typeof d.skills_count === "number" && (
                          <Badge variant="secondary" className="text-xs">навыков: {d.skills_count}</Badge>
                        )}
                      </span>
                    </span>
                    {onNavigate && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onNavigate("teacher")}
                        title={`Анализ дисциплины «${d.name}»`}
                        className="shrink-0 cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                      >
                        <Activity className="mr-1.5 size-3.5" />
                        Анализ
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            )
          )}
          {onNavigate && (
            <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-200 dark:border-slate-700 pt-4">
              <Button variant="outline" size="sm" onClick={() => onNavigate("teacher")} className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none">
                <Activity className="mr-1.5 size-3.5" />
                Преподавательский анализ
              </Button>
              <Button variant="outline" size="sm" onClick={() => onNavigate("taxonomy")} className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none">
                <BookOpen className="mr-1.5 size-3.5" />
                Таксономия
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

interface SelfData {
  profile: string;
  target_level: string;
  skills: string[];
  user_added: string[];
  competencies?: string[];
}

interface KrmSkill {
  text: string;
  student_has: boolean;
}

interface KrmCompetency {
  code: string;
  skills: KrmSkill[];
}

interface KrmDiscipline {
  discipline: string;
  competencies: KrmCompetency[];
}

interface KrmData {
  direction: string;
  direction_name: string;
  profile: string;
  disciplines: KrmDiscipline[];
  counts: { krm: number; student: number; overlap: number; merged: number; student_only: number };
}

const norm = (s: string) => s.toLowerCase().trim();

function looselyMatches(a: string, b: string): boolean {
  const x = norm(a);
  const y = norm(b);
  if (!x || !y) return false;
  if (x === y) return true;
  return x.length > 3 && y.length > 3 && (x.includes(y) || y.includes(x));
}

/** Полная страница профиля студента: identity + KPI + свои навыки + drill-down по КРМ. */
export function StudentProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  const auth = useAuth();
  const name = displayName ?? auth.name ?? undefined;
  const login = email ?? auth.username ?? undefined;

  const [self, setSelf] = useState<SelfData | null>(null);
  const [selfLoading, setSelfLoading] = useState(true);
  const [selfError, setSelfError] = useState<string | null>(null);
  const [krm, setKrm] = useState<KrmData | null>(null);
  const [krmLoading, setKrmLoading] = useState(true);
  const [krmError, setKrmError] = useState<string | null>(null);
  const [dirCode, setDirCode] = useState("09.03.02");
  const [openDisc, setOpenDisc] = useState<Record<string, boolean>>({});
  const [newSkill, setNewSkill] = useState("");
  const [saving, setSaving] = useState(false);
  const [opError, setOpError] = useState<string | null>(null);

  const loadKrm = async (profileName: string, dir: string) => {
    setKrmLoading(true);
    setKrmError(null);
    try {
      const r = await apiFetch(
        `/api/krm/student/competencies?direction=${encodeURIComponent(dir)}&profile=${encodeURIComponent(profileName)}`
      );
      if (!r.ok) throw new Error(`Не удалось загрузить компетенции (HTTP ${r.status})`);
      const d = (await r.json()) as KrmData;
      setKrm(d);
    } catch (e: any) {
      setKrmError(e?.message || "Ошибка загрузки компетенций");
      setKrm(null);
    } finally {
      setKrmLoading(false);
    }
  };

  const loadAll = async () => {
    setSelfLoading(true);
    setSelfError(null);
    setKrmLoading(true);
    setKrmError(null);
    try {
      const r = await apiFetch("/api/profiles/self");
      if (!r.ok) throw new Error(`Не удалось загрузить профиль (HTTP ${r.status})`);
      const d = (await r.json()) as SelfData;
      setSelf(d);
      const profileName = d.profile || "base";
      let dir = "09.03.02";
      try {
        const rd = await apiFetch("/api/krm/directions");
        if (rd.ok) {
          const dd = await rd.json();
          const first = dd?.directions?.[0]?.dir_code;
          if (typeof first === "string" && first) dir = first;
        }
      } catch { /* keep fallback */ }
      setDirCode(dir);
      await loadKrm(profileName, dir);
    } catch (e: any) {
      setSelfError(e?.message || "Ошибка загрузки профиля");
      setKrmLoading(false);
    } finally {
      setSelfLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      await loadAll();
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const patchSelf = async (body: Record<string, unknown>): Promise<SelfData> => {
    const r = await apiFetch("/api/profiles/self", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error((d as any).detail || "Не удалось сохранить");
    }
    return (await r.json()) as SelfData;
  };

  const handleAddSkill = async (raw?: string) => {
    const text = (raw ?? newSkill).trim();
    if (!text || saving || !self) return;
    const backup = self;
    setSaving(true);
    setOpError(null);
    const already = self.skills.some((s) => norm(s) === norm(text));
    if (!already) {
      setSelf({ ...self, skills: [...self.skills, text], user_added: [...(self.user_added || []), text] });
    }
    if (!raw) setNewSkill("");
    try {
      const updated = await patchSelf({ add_skills: [text] });
      setSelf(updated);
      await loadKrm(updated.profile || backup.profile || "base", dirCode);
    } catch (e: any) {
      setSelf(backup);
      if (!raw) setNewSkill(text);
      setOpError(e?.message || "Не удалось добавить навык");
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveSkill = async (text: string) => {
    if (saving || !self) return;
    const backup = self;
    setSaving(true);
    setOpError(null);
    setSelf({
      ...self,
      skills: self.skills.filter((s) => norm(s) !== norm(text)),
      user_added: (self.user_added || []).filter((s) => norm(s) !== norm(text)),
    });
    try {
      const updated = await patchSelf({ remove_skills: [text] });
      setSelf(updated);
      await loadKrm(updated.profile || backup.profile || "base", dirCode);
    } catch (e: any) {
      setSelf(backup);
      setOpError(e?.message || "Не удалось удалить навык");
    } finally {
      setSaving(false);
    }
  };

  const discStats = useMemo(() => {
    if (!krm) return [];
    return krm.disciplines.map((d) => {
      const total = d.competencies.reduce((a, c) => a + c.skills.length, 0);
      const has = d.competencies.reduce((a, c) => a + c.skills.filter((s) => s.student_has).length, 0);
      return { name: d.discipline, total, has, missing: total - has };
    });
  }, [krm]);

  const topGaps = useMemo(
    () => [...discStats].filter((d) => d.missing > 0).sort((a, b) => b.missing - a.missing).slice(0, 5),
    [discStats]
  );

  const missingPool = useMemo(() => {
    if (!krm) return [];
    const out: Array<{ text: string; code: string; discipline: string }> = [];
    for (const d of krm.disciplines) {
      for (const c of d.competencies) {
        for (const s of c.skills) {
          if (!s.student_has) {
            out.push({ text: s.text, code: c.code, discipline: d.discipline });
            if (out.length >= 8) return out;
          }
        }
      }
    }
    return out;
  }, [krm]);

  const skillsCount = self?.skills.length ?? 0;
  const krmTotal = krm?.counts.krm ?? 0;
  const overlap = krm?.counts.overlap ?? 0;
  const discsCount = krm?.disciplines.length ?? 0;
  const owned = new Set((self?.user_added || []).map((s) => norm(s)));
  const loading = selfLoading || krmLoading;

  return (
    <div className="space-y-6">
      {/* Identity header */}
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg text-slate-900 dark:text-slate-100">
            <User className="size-5 text-emerald-600 dark:text-emerald-400" />
            {name || "Мой профиль"}
          </CardTitle>
          {(login || self) && (
            <p className="text-sm text-slate-600 dark:text-slate-400">
              {login || "—"}
              {self ? ` · уровень ${self.target_level} · профиль ${self.profile}` : ""}
            </p>
          )}
        </CardHeader>
        <CardContent>
          {selfLoading && (
            <div className="flex items-center gap-3 animate-pulse" aria-label="Загрузка профиля">
              <div className="size-10 rounded-full bg-slate-200 dark:bg-slate-700" />
              <div className="space-y-2 flex-1">
                <div className="h-3 w-40 rounded bg-slate-200 dark:bg-slate-700" />
                <div className="h-3 w-24 rounded bg-slate-200 dark:bg-slate-700" />
              </div>
            </div>
          )}
          {selfError && (
            <p className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
              <AlertCircle className="size-4 shrink-0" />
              {selfError}
              <Button variant="outline" size="sm" onClick={() => void loadAll()} className="ml-2 cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none">
                Повторить
              </Button>
            </p>
          )}
        </CardContent>
      </Card>

      {/* KPI cards row */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2" aria-label="Загрузка показателей">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-3 animate-pulse">
              <div className="h-6 w-12 rounded bg-slate-200 dark:bg-slate-700 mx-auto" />
              <div className="mt-2 h-3 w-20 rounded bg-slate-200 dark:bg-slate-700 mx-auto" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {[
            { label: "Навыки", value: self ? String(skillsCount) : "—" },
            { label: "Компетенции КРМ", value: krm ? String(krmTotal) : "—" },
            { label: "Совпало", value: krm ? String(overlap) : "—" },
            { label: "Дисциплин", value: krm ? String(discsCount) : "—" },
          ].map((k) => (
            <div
              key={k.label}
              className="rounded-lg border border-slate-200 bg-white p-3 text-center transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900"
            >
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{k.value}</div>
              <div className="text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">{k.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Мои навыки */}
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg text-slate-900 dark:text-slate-100">
            <GraduationCap className="size-5 text-emerald-600 dark:text-emerald-400" />
            Мои навыки{self ? ` (${skillsCount})` : ""}
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">
            Свои навыки хранятся в вашем профиле — добавляйте и удаляйте
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {selfLoading && (
            <div className="flex flex-wrap gap-1.5 animate-pulse" aria-label="Загрузка навыков">
              {[0, 1, 2, 3, 4].map((i) => (
                <div key={i} className="h-6 w-20 rounded-md bg-slate-200 dark:bg-slate-700" />
              ))}
            </div>
          )}
          {selfError && !selfLoading && (
            <p className="text-sm text-red-600 dark:text-red-400">{selfError}</p>
          )}
          {self && !selfLoading && (
            <>
              {skillsCount === 0 ? (
                <p className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 px-3 py-4 text-center text-sm text-slate-600 dark:text-slate-400">
                  навыков пока нет — добавьте первый ниже
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {self.skills.map((s) => {
                    const mine = owned.has(norm(s));
                    return (
                      <Badge
                        key={s}
                        variant={mine ? "default" : "secondary"}
                        className="text-xs gap-1"
                        title={mine ? "Добавлен вами — можно удалить" : "Из программы — удалить нельзя"}
                      >
                        {s}
                        {mine && (
                          <button
                            onClick={() => void handleRemoveSkill(s)}
                            disabled={saving}
                            className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-red-300 focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:outline-none"
                            aria-label={`Удалить ${s}`}
                          >
                            <X className="size-3" />
                          </button>
                        )}
                      </Badge>
                    );
                  })}
                </div>
              )}
              <div className="flex gap-2">
                <Input
                  value={newSkill}
                  onChange={(e) => setNewSkill(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newSkill.trim()) void handleAddSkill();
                  }}
                  placeholder="Новый навык, Enter — добавить"
                  aria-label="Новый навык"
                  className="h-10 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                />
                <Button
                  onClick={() => void handleAddSkill()}
                  disabled={saving || !newSkill.trim()}
                  className="h-10 shrink-0 cursor-pointer bg-emerald-700 text-white hover:bg-emerald-800 transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                  aria-label="Добавить навык"
                >
                  {saving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
                </Button>
              </div>
              {opError && <p className="text-sm text-red-600 dark:text-red-400">{opError}</p>}
            </>
          )}
        </CardContent>
      </Card>

      {/* Мои компетенции — drill-down */}
      <Card className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg text-slate-900 dark:text-slate-100">
            <BookOpen className="size-5 text-indigo-600 dark:text-indigo-400" />
            Мои компетенции
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">
            Программа {krm?.direction_name || dirCode}: что совпало, чего не хватает
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {krmLoading && (
            <div className="space-y-2" aria-label="Загрузка компетенций">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg border border-slate-200 dark:border-slate-700 p-3 animate-pulse">
                  <div className="h-4 w-2/3 rounded bg-slate-200 dark:bg-slate-700" />
                  <div className="mt-2 flex gap-1.5">
                    <div className="h-5 w-16 rounded bg-slate-200 dark:bg-slate-700" />
                    <div className="h-5 w-24 rounded bg-slate-200 dark:bg-slate-700" />
                  </div>
                </div>
              ))}
            </div>
          )}
          {krmError && !krmLoading && (
            <p className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
              <AlertCircle className="size-4 shrink-0" />
              {krmError}
              <Button
                variant="outline"
                size="sm"
                onClick={() => void loadKrm(self?.profile || "base", dirCode)}
                className="ml-2 cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              >
                Повторить
              </Button>
            </p>
          )}
          {krm && !krmLoading && (
            <>
              {krm.disciplines.length === 0 ? (
                <p className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 px-3 py-4 text-center text-sm text-slate-600 dark:text-slate-400">
                  нет данных — добавьте навыки выше
                </p>
              ) : (
                <>
                  {topGaps.length > 0 && (
                    <div className="rounded-lg border border-slate-200 dark:border-slate-700 p-3">
                      <div className="text-sm font-medium text-slate-900 dark:text-slate-100">
                        Топ пробелов ({topGaps.length})
                      </div>
                      <ul className="mt-1.5 space-y-1 text-sm text-slate-600 dark:text-slate-400">
                        {topGaps.map((g) => (
                          <li key={g.name} className="flex items-center justify-between gap-2">
                            <span className="truncate">{g.name}</span>
                            <Badge variant="secondary" className="text-[11px] shrink-0">
                              не хватает {g.missing}/{g.total}
                            </Badge>
                          </li>
                        ))}
                      </ul>
                      {missingPool.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {missingPool.slice(0, 4).map((m) => (
                            <span
                              key={`${m.discipline}::${m.code}::${m.text}`}
                              className="inline-flex max-w-full items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700 transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
                            >
                              <span className="break-words">{m.text}</span>
                              <button
                                onClick={() => void handleAddSkill(m.text)}
                                disabled={saving}
                                className="cursor-pointer rounded p-0.5 text-slate-400 transition-colors duration-200 hover:bg-slate-100 hover:text-emerald-700 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none dark:hover:bg-slate-800 dark:hover:text-emerald-300"
                                aria-label={`Добавить «${m.text}»`}
                              >
                                <Plus className="size-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  <div className="space-y-2">
                    {krm.disciplines.map((d) => {
                      const st = discStats.find((s) => s.name === d.discipline);
                      const total = st?.total ?? 0;
                      const has = st?.has ?? 0;
                      const open = !!openDisc[d.discipline];
                      const matched = d.competencies.flatMap((c) => c.skills.filter((s) => s.student_has));
                      const missing = d.competencies.flatMap((c) =>
                        c.skills.filter((s) => !s.student_has).map((s) => ({ ...s, code: c.code }))
                      );
                      return (
                        <div key={d.discipline} className="rounded-lg border border-slate-200 bg-white transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900">
                          <button
                            onClick={() => setOpenDisc((p) => ({ ...p, [d.discipline]: !p[d.discipline] }))}
                            className="flex w-full cursor-pointer items-center justify-between gap-2 px-3 py-2.5 text-left transition-colors duration-200 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none dark:hover:bg-slate-800/50"
                            aria-expanded={open}
                          >
                            <span className="truncate text-sm font-medium text-slate-900 dark:text-slate-100">
                              {d.discipline}
                            </span>
                            <span className="flex shrink-0 items-center gap-2">
                              <Badge variant={has > 0 ? "default" : "secondary"} className="text-[11px]">
                                {has}/{total}
                              </Badge>
                              <ChevronDown className={`size-4 text-slate-400 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                            </span>
                          </button>
                          {open && (
                            <div className="space-y-2 px-3 pb-3">
                              {total === 0 && (
                                <p className="text-xs text-slate-600 dark:text-slate-400">
                                  нет данных — добавьте навыки выше
                                </p>
                              )}
                              {matched.length > 0 && (
                                <div>
                                  <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">
                                    Совпало ({matched.length})
                                  </div>
                                  <div className="flex flex-wrap gap-1.5">
                                    {matched.map((s, i) => (
                                      <span
                                        key={i}
                                        title="Есть у вас"
                                        className="inline-flex max-w-full items-center gap-1 rounded-md border border-emerald-200 bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-900 transition-colors duration-200 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
                                      >
                                        <CheckCircle2 className="size-3 shrink-0" />
                                        <span className="break-words">{s.text}</span>
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {missing.length > 0 && (
                                <div>
                                  <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-600 dark:text-slate-400">
                                    Не хватает ({missing.length})
                                  </div>
                                  <div className="flex flex-wrap gap-1.5">
                                    {missing.map((s, i) => (
                                      <span
                                        key={i}
                                        title="Нет у вас — нажмите +, чтобы добавить"
                                        className="inline-flex max-w-full items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700 transition-colors duration-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
                                      >
                                        <span className="break-words">{s.text}</span>
                                        <button
                                          onClick={() => void handleAddSkill(s.text)}
                                          disabled={saving}
                                          className="cursor-pointer rounded p-0.5 text-slate-400 transition-colors duration-200 hover:bg-slate-100 hover:text-emerald-700 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none dark:hover:bg-slate-800 dark:hover:text-emerald-300"
                                          aria-label={`Добавить «${s.text}»`}
                                        >
                                          <Plus className="size-3" />
                                        </button>
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
              <div>
                <Button
                  variant="outline"
                  onClick={() => onNavigate("data")}
                  className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
                >
                  <Database className="mr-2 size-4" />
                  Открыть результаты
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
