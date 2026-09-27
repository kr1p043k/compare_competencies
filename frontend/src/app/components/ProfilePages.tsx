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
  GraduationCap,  Cpu,  Layers,  Search,
  Database,
  AlertCircle,
} from "lucide-react";
import { apiFetch, useAuth } from "../../lib/auth"; import { Label } from "./ui/label"; import { Textarea } from "./ui/textarea"; import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

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
  technologies?: string[];
}

interface CustomOptionCompetency {
  code: string;
  title?: string;
  skills_count?: number;
}

interface CustomOptions {
  competencies: CustomOptionCompetency[];
  technologies_suggest: string[];
}

interface NewCompetencyDraft {
  code: string;
  title: string;
  knowledge: string;
  abilities: string;
  skills: string;
}

interface CreatedCustomProfile {
  name: string;
  target_level: string;
  competencies: number;
  created_new: number;
  technologies: number;
}

  const CUSTOM_NAME_RE = /^[a-z0-9_-]{2,40}$/;

const NO_BASE = "__none__";

const splitLines = (s: string): string[] =>
  s.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);

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

  // --- My technologies: chips + suggest + optimistic PATCH add_technologies/remove_technologies ---
  const [techInput, setTechInput] = useState("");
  const [techSaving, setTechSaving] = useState(false);
  const [techError, setTechError] = useState<string | null>(null);
  // --- My profiles: GET custom/options + POST custom (404-safe, inline errors) ---
  const [customOptions, setCustomOptions] = useState<CustomOptions | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [baseProfiles, setBaseProfiles] = useState<string[]>(["base"]);
  const [cpName, setCpName] = useState("");
  const [cpLevel, setCpLevel] = useState("middle");
  const [cpBase, setCpBase] = useState(NO_BASE);
  const [compSearch, setCompSearch] = useState("");
  const [compCodes, setCompCodes] = useState<string[]>([]);
  const [draftCode, setDraftCode] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const [draftKnowledge, setDraftKnowledge] = useState("");
  const [draftAbilities, setDraftAbilities] = useState("");
  const [draftSkills, setDraftSkills] = useState("");
  const [newComps, setNewComps] = useState<NewCompetencyDraft[]>([]);
  const [cpTech, setCpTech] = useState("");
  const [creating, setCreating] = useState(false);
  const [triedCreate, setTriedCreate] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createdProfiles, setCreatedProfiles] = useState<CreatedCustomProfile[]>([]);

  const techList = useMemo(() => self?.technologies ?? [], [self]);

  const techSuggestions = useMemo(() => {
    const pool = customOptions?.technologies_suggest ?? [];
    return pool.filter((t) => !techList.some((x) => norm(x) === norm(t))).slice(0, 10);
  }, [customOptions, techList]);

  const filteredCustomComps = useMemo(() => {
    const all = customOptions?.competencies ?? [];
    const q = norm(compSearch);
    if (!q) return all.slice(0, 200);
    return all.filter((c) => norm(c.code).includes(q) || norm(c.title ?? "").includes(q)).slice(0, 200);
  }, [customOptions, compSearch]);

  const cpNameNorm = cpName.trim().toLowerCase();
  const cpNameValid = CUSTOM_NAME_RE.test(cpNameNorm);
  const cpTechList = useMemo(
    () => cpTech.split(",").map((t) => t.trim()).filter((t) => t.length > 0),
    [cpTech]
  );
  const totalComps = compCodes.length + newComps.length;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch("/api/profiles/custom/options");
        if (!r.ok) throw new Error("HTTP " + r.status);
        const d = await r.json();
        if (cancelled) return;
        setCustomOptions({
          competencies: Array.isArray(d.competencies) ? d.competencies : [],
          technologies_suggest: Array.isArray(d.technologies_suggest) ? d.technologies_suggest : [],
        });
      } catch (e: any) {
        if (!cancelled) setOptionsError("Не удалось загрузить опции своих профилей (" + (e?.message || "Не удалось загрузить опции своих профилейB") + ")");
      } finally {
        if (!cancelled) setOptionsLoading(false);
      }
      try {
        const r = await apiFetch("/api/profiles");
        if (r.ok) {
          const d = await r.json();
          if (!cancelled && Array.isArray(d?.profiles) && d.profiles.length > 0) setBaseProfiles(d.profiles);
        }
      } catch {
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const patchTechnologies = async (body: { add_technologies?: string[]; remove_technologies?: string[] }): Promise<string[] | null> => {
    const r = await apiFetch("/api/profiles/self", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error((d as any).detail || "Не удалось сохранить технологии");
    }
    const updated = await r.json();
    return Array.isArray((updated as any).technologies) ? (updated as any).technologies : null;
  };

  const handleAddTech = async (rawTech?: string) => {
    const text = (rawTech ?? techInput).trim();
    if (!text || techSaving || !self) return;
    if (techList.some((t) => norm(t) === norm(text))) {
      if (!rawTech) setTechInput("");
      return;
    }
    const backup = techList;
    setTechSaving(true);
    setTechError(null);
    setSelf({ ...self, technologies: [...backup, text] });
    if (!rawTech) setTechInput("");
    try {
      const server = await patchTechnologies({ add_technologies: [text] });
      if (server !== null) setSelf((prev) => (prev ? { ...prev, technologies: server } : prev));
    } catch (e: any) {
      setSelf((prev) => (prev ? { ...prev, technologies: backup } : prev));
      if (!rawTech) setTechInput(text);
      setTechError(e?.message || "Не удалось добавить технологию");
    } finally {
      setTechSaving(false);
    }
  };

  const handleRemoveTech = async (text: string) => {
    if (techSaving || !self) return;
    const backup = techList;
    setTechSaving(true);
    setTechError(null);
    setSelf({ ...self, technologies: backup.filter((t) => norm(t) !== norm(text)) });
    try {
      const server = await patchTechnologies({ remove_technologies: [text] });
      if (server !== null) setSelf((prev) => (prev ? { ...prev, technologies: server } : prev));
    } catch (e: any) {
      setSelf((prev) => (prev ? { ...prev, technologies: backup } : prev));
      setTechError(e?.message || "Не удалось убрать технологию");
    } finally {
      setTechSaving(false);
    }
  };

  const toggleCompCode = (code: string) => {
    setCompCodes((prev) => (prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]));
  };

  const addDraft = () => {
    const code = draftCode.trim();
    if (!code) {
      setCreateError("Укажите код новой компетенции");
      return;
    }
    if (compCodes.includes(code) || newComps.some((d) => d.code === code)) {
      setCreateError("Компетенция" + code + " уже добавлена");
      return;
    }
    setCreateError(null);
    setNewComps((prev) => [...prev, { code: code, title: draftTitle.trim(), knowledge: draftKnowledge, abilities: draftAbilities, skills: draftSkills }]);
    setDraftCode("");
    setDraftTitle("");
    setDraftKnowledge("");
    setDraftAbilities("");
    setDraftSkills("");
  };

  const removeDraft = (code: string) => {
    setNewComps((prev) => prev.filter((d) => d.code !== code));
  };

  const handleCreateProfile = async () => {
    setTriedCreate(true);
    if (!cpNameValid) {
      setCreateError("Название: латиница, цифры и подчёркивание, 2-32 символа");
      return;
    }
    if (totalComps < 1) {
      setCreateError("Выберите хотя бы одну компетенцию или добавьте новую");
      return;
    }
    setCreating(true);
    setCreateError(null);
    const payload = {
      name: cpNameNorm,
      target_level: cpLevel,
      ...(cpBase !== NO_BASE ? { base: cpBase } : {}),
      competency_codes: compCodes,
      new_competencies: newComps.map((d) => ({
        code: d.code,
        ...(d.title ? { title: d.title } : {}),
        knowledge: splitLines(d.knowledge),
        abilities: splitLines(d.abilities),
        skills: splitLines(d.skills),
      })),
      technologies: cpTechList,
    };
    try {
      const r = await apiFetch("/api/profiles/custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error((d as any).detail || "Не удалось создать профиль (HTTP " + r.status + ")");
      }
      const d = await r.json();
      setCreatedProfiles((prev) => [
        ...prev,
        {
          name: (d as any).profile ?? (d as any).name ?? cpNameNorm,
          target_level: (d as any).target_level ?? cpLevel,
          competencies: (d as any).competencies_count ?? compCodes.length,
          created_new: (d as any).new_competencies_count ?? newComps.length,
          technologies: (d as any).technologies_count ?? cpTechList.length,
        },
      ]);
      setCpName("");
      setCpLevel("middle");
      setCpBase(NO_BASE);
      setCompSearch("");
      setCompCodes([]);
      setNewComps([]);
      setCpTech("");
      setTriedCreate(false);
    } catch (e: any) {
      setCreateError(e?.message || "Не удалось загрузить опции своих профилей0");
    } finally {
      setCreating(false);
    }
  };

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

      <Card id="my-technologies" className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg text-slate-900 dark:text-slate-100">
            <Cpu className="size-5 text-emerald-600 dark:text-emerald-400" />
            Мои технологии{self && !selfLoading ? " (" + techList.length + ")" : ""}
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">
            Стек, с которым вы работаете: добавляйте новое и убирайте устаревшее
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {selfLoading && (
            <div className="flex flex-wrap gap-1.5 animate-pulse" aria-label="Загрузка технологий">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-6 w-20 rounded-md bg-slate-200 dark:bg-slate-700" />
              ))}
            </div>
          )}
          {self && !selfLoading && (
            <>
              {techList.length === 0 ? (
                <p className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 px-3 py-4 text-center text-sm text-slate-600 dark:text-slate-400">
                  Пока пусто — добавьте первую технологию ниже. Подсказки берутся из списка technologies_suggest.
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {techList.map((t) => (
                    <Badge key={t} variant="secondary" className="text-xs gap-1">
                      {t}
                      <button
                        onClick={() => void handleRemoveTech(t)}
                        disabled={techSaving}
                        className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-red-500 focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:outline-none"
                        aria-label={"Убрать " + t}
                      >
                        <X className="size-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
              <div className="flex gap-2">
                <Input
                  value={techInput}
                  onChange={(e) => setTechInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && techInput.trim()) void handleAddTech();
                  }}
                  placeholder="Например: Docker. Enter — добавить"
                  aria-label="Новая технология"
                  list="my-tech-suggest"
                  className="h-10 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                />
                <datalist id="my-tech-suggest">
                  {(customOptions?.technologies_suggest ?? []).map((s) => (
                    <option key={s} value={s} />
                  ))}
                </datalist>
                <Button
                  onClick={() => void handleAddTech()}
                  disabled={techSaving || !techInput.trim()}
                  className="h-10 shrink-0 cursor-pointer bg-emerald-700 text-white hover:bg-emerald-800 transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                  aria-label="Добавить технологию"
                >
                  {techSaving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
                </Button>
              </div>
              {techSuggestions.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {techSuggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => void handleAddTech(s)}
                      disabled={techSaving}
                      className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700 transition-colors duration-200 hover:border-emerald-300 hover:text-emerald-700 focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-emerald-700 dark:hover:text-emerald-300"
                      aria-label={"Добавить " + s}
                    >
                      <Plus className="size-3" />
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {techError && <p className="text-sm text-red-600 dark:text-red-400">{techError}</p>}
            </>
          )}
        </CardContent>
      </Card>

      <Card id="my-profiles" className="bg-white dark:bg-slate-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg text-slate-900 dark:text-slate-100">
            <Layers className="size-5 text-indigo-600 dark:text-indigo-400" />
            Мои профили{createdProfiles.length > 0 ? " (" + createdProfiles.length + ")" : ""}
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-400">
            Свои профили компетенций: соберите из готовых или опишите новые
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {optionsLoading && (
            <div className="space-y-2 animate-pulse" aria-label="Загрузка опций профилей">
              <div className="h-4 w-2/3 rounded bg-slate-200 dark:bg-slate-700" />
              <div className="h-9 rounded bg-slate-200 dark:bg-slate-700" />
            </div>
          )}
          {optionsError && (
            <p className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
              <AlertCircle className="size-4 shrink-0" />
              {optionsError}
            </p>
          )}
          {createdProfiles.length > 0 && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100">
                Создано в этой сессии ({createdProfiles.length})
              </div>
              {createdProfiles.map((cp) => (
                <div
                  key={cp.name}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 transition-colors duration-200 dark:border-emerald-800 dark:bg-emerald-950/40"
                >
                  <span className="flex items-center gap-2 text-sm font-medium text-emerald-900 dark:text-emerald-200">
                    <CheckCircle2 className="size-4 shrink-0" />
                    {cp.name}
                  </span>
                  <span className="flex flex-wrap gap-1.5">
                    <Badge variant="secondary" className="text-[11px]">{cp.target_level}</Badge>
                    <Badge variant="secondary" className="text-[11px]">компетенций: {cp.competencies}</Badge>
                    {cp.created_new > 0 && (
                      <Badge variant="secondary" className="text-[11px]">новых: {cp.created_new}</Badge>
                    )}
                    {cp.technologies > 0 && (
                      <Badge variant="secondary" className="text-[11px]">технологий: {cp.technologies}</Badge>
                    )}
                  </span>
                </div>
              ))}
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Сервер отдаёт только общий список профилей, поэтому здесь показаны профили, созданные в этой сессии.
              </p>
            </div>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="cp-name">Название профиля</Label>
              <Input
                id="cp-name"
                value={cpName}
                onChange={(e) => setCpName(e.target.value)}
                placeholder="my_backend"
                aria-label="Название профиля"
                className="h-10 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
              <p className="text-xs text-slate-500 dark:text-slate-400">Латиница, цифры и подчёркивание, 2-32 символа</p>
            </div>
            <div className="space-y-1.5">
              <Label>Целевой уровень</Label>
              <Select value={cpLevel} onValueChange={setCpLevel}>
                <SelectTrigger className="h-10" aria-label="Целевой уровень">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="junior">Junior</SelectItem>
                  <SelectItem value="middle">Middle</SelectItem>
                  <SelectItem value="senior">Senior</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label>База</Label>
            <Select value={cpBase} onValueChange={setCpBase}>
              <SelectTrigger className="h-10 w-full sm:w-64" aria-label="Базовый профиль">
                <SelectValue placeholder="Без базы" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">Без базы</SelectItem>
                {baseProfiles.map((b) => (
                  <SelectItem key={b} value={b}>
                    {b}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
            <div className="flex items-center justify-between gap-2">
              <Label>Компетенции</Label>
              <Badge variant="secondary" className="text-[11px]">выбрано: {compCodes.length}</Badge>
            </div>

            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
              <Input
                value={compSearch}
                onChange={(e) => setCompSearch(e.target.value)}
                placeholder="Поиск по коду или названию"
                aria-label="Поиск компетенций"
                className="h-9 pl-8 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
            </div>
            {filteredCustomComps.length === 0 ? (
              <p className="rounded-md border border-dashed border-slate-300 dark:border-slate-700 px-3 py-3 text-center text-xs text-slate-500 dark:text-slate-400">
                Нет компетенций для выбора. Проверьте поиск или добавьте новую ниже.
              </p>
            ) : (
              <ul className="max-h-56 space-y-1 overflow-y-auto pr-1">
                {filteredCustomComps.map((c) => {
                  const checked = compCodes.includes(c.code);
                  return (
                    <li key={c.code}>
                      <label className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 transition-colors duration-200 hover:bg-slate-50 dark:hover:bg-slate-800/60">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleCompCode(c.code)}
                          className="mt-1 size-4 shrink-0 cursor-pointer accent-indigo-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded"
                          aria-label={c.code}
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-medium text-slate-900 dark:text-slate-100">{c.code}</span>
                          {c.title ? (
                            <span className="block truncate text-xs text-slate-600 dark:text-slate-400">{c.title}</span>
                          ) : null}
                        </span>
                        {typeof c.skills_count === "number" ? (
                          <Badge variant="secondary" className="shrink-0 text-[11px]">
                            {c.skills_count}
                          </Badge>
                        ) : null}
                      </label>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          <div className="space-y-2 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
            <Label>Новая компетенция</Label>

            <div className="grid gap-2 sm:grid-cols-2">
              <Input
                value={draftCode}
                onChange={(e) => setDraftCode(e.target.value)}
                placeholder="Код, например ПК-5"
                aria-label="Код компетенции"
                className="h-10 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
              <Input
                value={draftTitle}
                onChange={(e) => setDraftTitle(e.target.value)}
                placeholder="Название (необязательно)"
                aria-label="Название компетенции"
                className="h-10 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
            </div>
            <div className="grid gap-2 md:grid-cols-3">
              <Textarea
                value={draftKnowledge}
                onChange={(e) => setDraftKnowledge(e.target.value)}
                placeholder="Знания — по одному на строку"
                aria-label="Знания"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
              <Textarea
                value={draftAbilities}
                onChange={(e) => setDraftAbilities(e.target.value)}
                placeholder="Умения — по одному на строку"
                aria-label="Умения"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
              <Textarea
                value={draftSkills}
                onChange={(e) => setDraftSkills(e.target.value)}
                placeholder="Навыки — по одному на строку"
                aria-label="Навыки"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
              />
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={addDraft}
              className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
            >
              <Plus className="mr-1.5 size-3.5" />
              Добавить компетенцию
            </Button>
            {newComps.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {newComps.map((d) => (
                  <Badge key={d.code} variant="default" className="text-xs gap-1">
                    {d.code}
                    <button
                      onClick={() => removeDraft(d.code)}
                      className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-red-300 focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:outline-none"
                      aria-label={"Убрать " + d.code}
                    >
                      <X className="size-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cp-tech">Технологии (через запятую)</Label>
            <Input
              id="cp-tech"
              value={cpTech}
              onChange={(e) => setCpTech(e.target.value)}
              placeholder="Python, Docker, PostgreSQL"
              aria-label="Технологии профиля"
              className="h-10 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
            />
          </div>
          {!cpNameValid && cpName.length > 0 && (
            <p className="text-sm text-red-600 dark:text-red-400">Название: латиница, цифры и подчёркивание, 2-32 символа</p>
          )}
          {triedCreate && totalComps < 1 && (
            <p className="text-sm text-amber-700 dark:text-amber-300">Выберите хотя бы одну компетенцию или добавьте новую</p>
          )}
          {createError && <p className="text-sm text-red-600 dark:text-red-400">{createError}</p>}
          <Button
            onClick={() => void handleCreateProfile()}
            disabled={creating}
            className="h-10 cursor-pointer bg-indigo-700 text-white hover:bg-indigo-800 transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:outline-none"
          >
            {creating ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Plus className="mr-2 size-4" />}
            Создать профиль
          </Button>
        </CardContent>
      </Card>
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
