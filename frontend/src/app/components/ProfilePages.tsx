import { useEffect, useMemo, useState } from "react";
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
  GraduationCap, Cpu, Layers, Search,
  Database,
} from "lucide-react";
import { useAuth } from "../../lib/auth";
import { TwoFactorCard } from "./TwoFactorCard";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  PageHeader,
  Skeleton,
  StatCard,
  inputClass,
} from "../../design";
import { AdminSecurity } from "./AdminSecurity";
import { cn } from "./ui/utils";

type AccountProps = { displayName?: string; email?: string };

function AccountCard({ displayName, email, roleLabel }: AccountProps & { roleLabel: string }) {
  return (
    <Card>
      <CardHeader title="Учётная запись" description={roleLabel} />
      <CardBody className="space-y-1 text-sm text-foreground">
        <p>
          <span className="text-muted-foreground">ФИО: </span>
          {displayName || "-"}
        </p>
        <p>
          <span className="text-muted-foreground">Email: </span>
          {email || "-"}
        </p>
      </CardBody>
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
        const d = await request<Record<string, unknown>>("/api/admin/monitoring/summary");
        if (!cancelled) setSum(d as never);
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
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, дисциплины и уведомления администратора."
      />
      <AccountCard displayName={displayName} email={email} roleLabel="Администратор" />
      <TwoFactorCard />
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Activity className="size-5 text-primary" />
              Живая сводка
            </span>
          }
          description="Активность, ошибки и свежесть данных - прочерк при недоступности"
        />
        <CardBody>
          {sumLoading ? (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2" aria-label="Загрузка сводки">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg border border-border p-3">
                  <Skeleton className="h-6 w-12 mx-auto" />
                  <Skeleton className="mt-2 h-3 w-20 mx-auto" />
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
              <StatCard
                className="p-3"
                label="Сессии активны"
                value={typeof sum?.sessions_active === "number" ? sum.sessions_active : "-"}
              />
              <StatCard
                className="p-3"
                label="Ошибки за час"
                value={errorsTotal === null ? "-" : errorsTotal}
              />
              <StatCard
                className="col-span-2 lg:col-span-1 p-3"
                label={`Вакансии${sum?.freshness?.date_from ? ` · ${sum.freshness.date_from}-${sum.freshness.date_to ?? "…"}` : ""}`}
                value={
                  typeof sum?.freshness?.vacancies === "number" ? sum.freshness.vacancies.toLocaleString("ru-RU") : "-"
                }
              />
            </div>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Shield className="size-5 text-primary" />
              Администрирование
            </span>
          }
          description="Быстрые ссылки на административные разделы"
        />
        <CardBody className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("admin")}
            iconLeft={<Users className="size-4" />}
          >
            Пользователи
          </Button>
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("logs")}
            iconLeft={<FileText className="size-4" />}
          >
            Логи
          </Button>
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("monitoring")}
            iconLeft={<Activity className="size-4" />}
          >
            Мониторинг
          </Button>
        </CardBody>
      </Card>
      <AdminSecurity />
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
    request<DisciplineRow[]>("/api/teacher/krm/disciplines?dir_code=09.03.02")
      .then((d) => { if (!cancelled) setDiscs(Array.isArray(d) ? d : []); })
      .catch((e: any) => { if (!cancelled) setError(e?.message || "Не удалось загрузить дисциплины"); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, дисциплины и уведомления преподавателя."
      />
      <AccountCard displayName={displayName} email={email} roleLabel="Преподаватель" />
      <TwoFactorCard />
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <BookOpen className="size-5 text-success" />
              Зона преподавания
            </span>
          }
          description="Дисциплины направления 09.03.02 (только просмотр)"
        />
        <CardBody>
          {discs === null && !error && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Загрузка…
            </p>
          )}
          {error && <ErrorState description={error} />}
          {discs !== null && !error && (
            discs.length === 0 ? (
              <EmptyState title="Дисциплины не найдены" />
            ) : (
              <ul className="divide-y divide-border">
                {discs.map((d) => (
                  <li key={d.name} className="py-2 flex items-center justify-between gap-3">
                    <span className="text-sm text-foreground min-w-0">
                      <span className="block truncate">{d.name}</span>
                      <span className="flex items-center gap-1.5 mt-1 flex-wrap">
                        {typeof d.semester === "number" && (
                          <Badge tone="neutral" className="text-xs">сем. {d.semester}</Badge>
                        )}
                        {typeof d.competencies_count === "number" && (
                          <Badge tone="neutral" className="text-xs">компетенций: {d.competencies_count}</Badge>
                        )}
                        {typeof d.skills_count === "number" && (
                          <Badge tone="neutral" className="text-xs">навыков: {d.skills_count}</Badge>
                        )}
                      </span>
                    </span>
                    {onNavigate && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => onNavigate("teacher")}
                        title={`Анализ дисциплины «${d.name}»`}
                        className="shrink-0 cursor-pointer"
                        iconLeft={<Activity className="size-3.5" />}
                      >
                        Анализ
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            )
          )}
          {onNavigate && (
            <div className="mt-4 flex flex-wrap gap-2 border-t border-border pt-4">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onNavigate("teacher")}
                className="cursor-pointer"
                iconLeft={<Activity className="size-3.5" />}
              >
                Преподавательский анализ
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onNavigate("taxonomy")}
                className="cursor-pointer"
                iconLeft={<BookOpen className="size-3.5" />}
              >
                Таксономия
              </Button>
            </div>
          )}
        </CardBody>
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
        const d = await request<{
          competencies?: CustomOptionCompetency[];
          technologies_suggest?: string[];
        }>("/api/profiles/custom/options");
        if (cancelled) return;
        setCustomOptions({
          competencies: Array.isArray(d.competencies) ? d.competencies : [],
          technologies_suggest: Array.isArray(d.technologies_suggest) ? d.technologies_suggest : [],
        });
      } catch (e: any) {
        if (!cancelled) setOptionsError("Не удалось загрузить опции своих профилей (" + (e?.message || "Не удалось загрузить опции своих профилей") + ")");
      } finally {
        if (!cancelled) setOptionsLoading(false);
      }
      try {
        const d = await request<{ profiles?: string[] }>("/api/profiles");
        if (!cancelled && Array.isArray(d?.profiles) && d.profiles.length > 0) {
          setBaseProfiles(d.profiles);
        }
      } catch (err) {
        reportError("ProfilePages:baseProfiles", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const patchTechnologies = async (body: { add_technologies?: string[]; remove_technologies?: string[] }): Promise<string[] | null> => {
    const updated = await request<{ technologies?: string[] }>("/api/profiles/self", {
      method: "PATCH",
      body,
    });
    return Array.isArray(updated.technologies) ? updated.technologies : null;
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
      setCreateError(`Компетенция ${code} уже добавлена`);
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
      const d = await request<{
        profile?: string;
        name?: string;
        target_level?: string;
        competencies_count?: number;
        new_competencies_count?: number;
        technologies_count?: number;
      }>("/api/profiles/custom", { method: "POST", body: payload });
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
      setCreateError(e?.message || "Не удалось создать профиль");
    } finally {
      setCreating(false);
    }
  };

  const loadKrm = async (profileName: string, dir: string) => {
    setKrmLoading(true);
    setKrmError(null);
    try {
      const d = await request<KrmData>(
        `/api/krm/student/competencies?direction=${encodeURIComponent(dir)}&profile=${encodeURIComponent(profileName)}`,
      );
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
      const d = await request<SelfData>("/api/profiles/self");
      setSelf(d);
      const profileName = d.profile || "base";
      let dir = "09.03.02";
      try {
        const dd = await request<{ directions?: { dir_code?: string }[] }>("/api/krm/directions");
        const first = dd?.directions?.[0]?.dir_code;
        if (typeof first === "string" && first) dir = first;
      } catch (err) {
        reportError("ProfilePages:krmDirections", err);
      }
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
    return request<SelfData>("/api/profiles/self", { method: "PATCH", body });
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
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, навыки и свои профили для сравнения с рынком."
      />
      {/* Identity header */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <User className="size-5 text-success" />
              {name || "Мой профиль"}
            </span>
          }
          description={
            login || self ? (
              <>
                {login || "-"}
                {self ? ` · уровень ${self.target_level} · профиль ${self.profile}` : ""}
              </>
            ) : undefined
          }
        />
        <CardBody>
          {selfLoading && (
            <div className="flex items-center gap-3" aria-label="Загрузка профиля">
              <Skeleton className="size-10 rounded-full" />
              <div className="space-y-2 flex-1">
                <Skeleton className="h-3 w-40" />
                <Skeleton className="h-3 w-24" />
              </div>
            </div>
          )}
          {selfError && <ErrorState description={selfError} onRetry={() => void loadAll()} />}
        </CardBody>
      </Card>

      <TwoFactorCard />

      {/* KPI cards row */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2" aria-label="Загрузка показателей">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="rounded-lg border border-border bg-card p-3">
              <Skeleton className="h-6 w-12 mx-auto" />
              <Skeleton className="mt-2 h-3 w-20 mx-auto" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {[
            { label: "Навыки", value: self ? String(skillsCount) : "-" },
            { label: "Компетенции КРМ", value: krm ? String(krmTotal) : "-" },
            { label: "Совпало", value: krm ? String(overlap) : "-" },
            { label: "Дисциплин", value: krm ? String(discsCount) : "-" },
          ].map((k) => (
            <StatCard key={k.label} className="p-3" label={k.label} value={k.value} />
          ))}
        </div>
      )}

      {/* Мои навыки */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <GraduationCap className="size-5 text-success" />
              Мои навыки{self ? ` (${skillsCount})` : ""}
            </span>
          }
          description="Свои навыки хранятся в вашем профиле - добавляйте и удаляйте"
        />
        <CardBody className="space-y-3">
          {selfLoading && (
            <div className="flex flex-wrap gap-1.5" aria-label="Загрузка навыков">
              {[0, 1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-6 w-20" />
              ))}
            </div>
          )}
          {selfError && !selfLoading && (
            <p className="text-sm text-destructive">{selfError}</p>
          )}
          {self && !selfLoading && (
            <>
              {skillsCount === 0 ? (
                <EmptyState title="Навыков пока нет" description="Добавьте первый ниже" />
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {self.skills.map((s) => {
                    const mine = owned.has(norm(s));
                    return (
                      <Badge
                        key={s}
                        tone={mine ? "primary" : "neutral"}
                        className="text-xs gap-1"
                        title={mine ? "Добавлен вами - можно удалить" : "Из программы - удалить нельзя"}
                      >
                        {s}
                        {mine && (
                          <button
                            onClick={() => void handleRemoveSkill(s)}
                            disabled={saving}
                            className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
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
                <input
                  value={newSkill}
                  onChange={(e) => setNewSkill(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newSkill.trim()) void handleAddSkill();
                  }}
                  placeholder="Новый навык, Enter - добавить"
                  aria-label="Новый навык"
                  className={inputClass}
                />
                <Button
                  variant="success"
                  onClick={() => void handleAddSkill()}
                  disabled={saving || !newSkill.trim()}
                  className="shrink-0 cursor-pointer"
                  aria-label="Добавить навык"
                >
                  {saving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
                </Button>
              </div>
              {opError && <p className="text-sm text-destructive">{opError}</p>}
            </>
          )}
        </CardBody>
      </Card>

      {/* Мои компетенции - drill-down */}

      <Card id="my-technologies">
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Cpu className="size-5 text-success" />
              Мои технологии{self && !selfLoading ? " (" + techList.length + ")" : ""}
            </span>
          }
          description="Стек, с которым вы работаете: добавляйте новое и убирайте устаревшее"
        />
        <CardBody className="space-y-3">
          {selfLoading && (
            <div className="flex flex-wrap gap-1.5" aria-label="Загрузка технологий">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-6 w-20" />
              ))}
            </div>
          )}
          {self && !selfLoading && (
            <>
              {techList.length === 0 ? (
                <EmptyState title="Пока пусто" description="Добавьте первую технологию ниже. Подсказки берутся из списка technologies_suggest." />
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {techList.map((t) => (
                    <Badge key={t} tone="neutral" className="text-xs gap-1">
                      {t}
                      <button
                        onClick={() => void handleRemoveTech(t)}
                        disabled={techSaving}
                        className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                        aria-label={"Убрать " + t}
                      >
                        <X className="size-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
              <div className="flex gap-2">
                <input
                  value={techInput}
                  onChange={(e) => setTechInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && techInput.trim()) void handleAddTech();
                  }}
                  placeholder="Например: Docker. Enter - добавить"
                  aria-label="Новая технология"
                  list="my-tech-suggest"
                  className={inputClass}
                />
                <datalist id="my-tech-suggest">
                  {(customOptions?.technologies_suggest ?? []).map((s) => (
                    <option key={s} value={s} />
                  ))}
                </datalist>
                <Button
                  variant="success"
                  onClick={() => void handleAddTech()}
                  disabled={techSaving || !techInput.trim()}
                  className="shrink-0 cursor-pointer"
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
                      className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border bg-card px-2 py-0.5 text-xs text-foreground transition-colors duration-200 hover:border-success/60 hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      aria-label={"Добавить " + s}
                    >
                      <Plus className="size-3" />
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {techError && <p className="text-sm text-destructive">{techError}</p>}
            </>
          )}
        </CardBody>
      </Card>

      <Card id="my-profiles">
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Layers className="size-5 text-primary" />
              Мои профили{createdProfiles.length > 0 ? " (" + createdProfiles.length + ")" : ""}
            </span>
          }
          description="Свои профили компетенций: соберите из готовых или опишите новые"
        />
        <CardBody className="space-y-4">
          {optionsLoading && (
            <div className="space-y-2" aria-label="Загрузка опций профилей">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-9" />
            </div>
          )}
          {optionsError && <ErrorState description={optionsError} />}
          {createdProfiles.length > 0 && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-foreground">
                Создано в этой сессии ({createdProfiles.length})
              </div>
              {createdProfiles.map((cp) => (
                <div
                  key={cp.name}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-success/30 bg-success-soft px-3 py-2 transition-colors duration-200"
                >
                  <span className="flex items-center gap-2 text-sm font-medium text-success">
                    <CheckCircle2 className="size-4 shrink-0" />
                    {cp.name}
                  </span>
                  <span className="flex flex-wrap gap-1.5">
                    <Badge tone="neutral" className="text-xs">{cp.target_level}</Badge>
                    <Badge tone="neutral" className="text-xs">компетенций: {cp.competencies}</Badge>
                    {cp.created_new > 0 && (
                      <Badge tone="neutral" className="text-xs">новых: {cp.created_new}</Badge>
                    )}
                    {cp.technologies > 0 && (
                      <Badge tone="neutral" className="text-xs">технологий: {cp.technologies}</Badge>
                    )}
                  </span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground">
                Сервер отдаёт только общий список профилей, поэтому здесь показаны профили, созданные в этой сессии.
              </p>
            </div>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="cp-name">Название профиля</Label>
              <input
                id="cp-name"
                value={cpName}
                onChange={(e) => setCpName(e.target.value)}
                placeholder="my_backend"
                aria-label="Название профиля"
                className={inputClass}
              />
              <p className="text-xs text-muted-foreground">Латиница, цифры и подчёркивание, 2-32 символа</p>
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
          <div className="space-y-2 rounded-lg border border-border p-3">
            <div className="flex items-center justify-between gap-2">
              <Label>Компетенции</Label>
              <Badge tone="neutral" className="text-xs">выбрано: {compCodes.length}</Badge>
            </div>

            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={compSearch}
                onChange={(e) => setCompSearch(e.target.value)}
                placeholder="Поиск по коду или названию"
                aria-label="Поиск компетенций"
                className={cn(inputClass, "h-9 pl-8")}
              />
            </div>
            {filteredCustomComps.length === 0 ? (
              <EmptyState title="Нет компетенций для выбора" description="Проверьте поиск или добавьте новую ниже" />
            ) : (
              <ul className="max-h-56 space-y-1 overflow-y-auto pr-1">
                {filteredCustomComps.map((c) => {
                  const checked = compCodes.includes(c.code);
                  return (
                    <li key={c.code}>
                      <label className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 transition-colors duration-200 hover:bg-muted">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleCompCode(c.code)}
                          className="mt-1 size-4 shrink-0 cursor-pointer accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
                          aria-label={c.code}
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-medium text-foreground">{c.code}</span>
                          {c.title ? (
                            <span className="block truncate text-xs text-muted-foreground">{c.title}</span>
                          ) : null}
                        </span>
                        {typeof c.skills_count === "number" ? (
                          <Badge tone="neutral" className="shrink-0 text-xs">
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
          <div className="space-y-2 rounded-lg border border-border p-3">
            <Label>Новая компетенция</Label>

            <div className="grid gap-2 sm:grid-cols-2">
              <input
                value={draftCode}
                onChange={(e) => setDraftCode(e.target.value)}
                placeholder="Код, например ПК-5"
                aria-label="Код компетенции"
                className={inputClass}
              />
              <input
                value={draftTitle}
                onChange={(e) => setDraftTitle(e.target.value)}
                placeholder="Название (необязательно)"
                aria-label="Название компетенции"
                className={inputClass}
              />
            </div>
            <div className="grid gap-2 md:grid-cols-3">
              <Textarea
                value={draftKnowledge}
                onChange={(e) => setDraftKnowledge(e.target.value)}
                placeholder="Знания - по одному на строку"
                aria-label="Знания"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              />
              <Textarea
                value={draftAbilities}
                onChange={(e) => setDraftAbilities(e.target.value)}
                placeholder="Умения - по одному на строку"
                aria-label="Умения"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              />
              <Textarea
                value={draftSkills}
                onChange={(e) => setDraftSkills(e.target.value)}
                placeholder="Навыки - по одному на строку"
                aria-label="Навыки"
                rows={3}
                className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              />
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={addDraft}
              className="cursor-pointer"
              iconLeft={<Plus className="size-3.5" />}
            >
              Добавить компетенцию
            </Button>
            {newComps.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {newComps.map((d) => (
                  <Badge key={d.code} tone="primary" className="text-xs gap-1">
                    {d.code}
                    <button
                      onClick={() => removeDraft(d.code)}
                      className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
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
            <input
              id="cp-tech"
              value={cpTech}
              onChange={(e) => setCpTech(e.target.value)}
              placeholder="Python, Docker, PostgreSQL"
              aria-label="Технологии профиля"
              className={inputClass}
            />
          </div>
          {!cpNameValid && cpName.length > 0 && (
            <p className="text-sm text-destructive">Название: латиница, цифры и подчёркивание, 2-32 символа</p>
          )}
          {triedCreate && totalComps < 1 && (
            <p className="text-sm text-warning">Выберите хотя бы одну компетенцию или добавьте новую</p>
          )}
          {createError && <p className="text-sm text-destructive">{createError}</p>}
          <Button
            variant="primary"
            onClick={() => void handleCreateProfile()}
            disabled={creating}
            className="cursor-pointer"
          >
            {creating ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Plus className="mr-2 size-4" />}
            Создать профиль
          </Button>
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <BookOpen className="size-5 text-primary" />
              Мои компетенции
            </span>
          }
          description={`Программа ${krm?.direction_name || dirCode}: что совпало, чего не хватает`}
        />
        <CardBody className="space-y-3">
          {krmLoading && (
            <div className="space-y-2" aria-label="Загрузка компетенций">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg border border-border p-3">
                  <Skeleton className="h-4 w-2/3" />
                  <div className="mt-2 flex gap-1.5">
                    <Skeleton className="h-5 w-16" />
                    <Skeleton className="h-5 w-24" />
                  </div>
                </div>
              ))}
            </div>
          )}
          {krmError && !krmLoading && (
            <ErrorState description={krmError} onRetry={() => void loadKrm(self?.profile || "base", dirCode)} />
          )}
          {krm && !krmLoading && (
            <>
              {krm.disciplines.length === 0 ? (
                <EmptyState title="Нет данных" description="Добавьте навыки выше" />
              ) : (
                <>
                  {topGaps.length > 0 && (
                    <div className="rounded-lg border border-border p-3">
                      <div className="text-sm font-medium text-foreground">
                        Топ пробелов ({topGaps.length})
                      </div>
                      <ul className="mt-1.5 space-y-1 text-sm text-muted-foreground">
                        {topGaps.map((g) => (
                          <li key={g.name} className="flex items-center justify-between gap-2">
                            <span className="truncate">{g.name}</span>
                            <Badge tone="neutral" className="text-xs shrink-0">
                              не хватает {g.missing}/{g.total}
                            </Badge>
                          </li>
                        ))}
                      </ul>
                      {missingPool.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {missingPool.slice(0, 4).map((m) => (
                            <Badge
                              key={`${m.discipline}::${m.code}::${m.text}`}
                              tone="neutral"
                              className="max-w-full text-xs"
                            >
                              <span className="break-words">{m.text}</span>
                              <button
                                onClick={() => void handleAddSkill(m.text)}
                                disabled={saving}
                                className="cursor-pointer rounded p-0.5 text-muted-foreground transition-colors duration-200 hover:bg-muted hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                                aria-label={`Добавить «${m.text}»`}
                              >
                                <Plus className="size-3" />
                              </button>
                            </Badge>
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
                        <div key={d.discipline} className="rounded-lg border border-border bg-card transition-colors duration-200">
                          <button
                            onClick={() => setOpenDisc((p) => ({ ...p, [d.discipline]: !p[d.discipline] }))}
                            className="flex w-full cursor-pointer items-center justify-between gap-2 px-3 py-2.5 text-left transition-colors duration-200 hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                            aria-expanded={open}
                          >
                            <span className="truncate text-sm font-medium text-foreground">
                              {d.discipline}
                            </span>
                            <span className="flex shrink-0 items-center gap-2">
                              <Badge tone={has > 0 ? "primary" : "neutral"} className="text-xs">
                                {has}/{total}
                              </Badge>
                              <ChevronDown className={`size-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                            </span>
                          </button>
                          {open && (
                            <div className="space-y-2 px-3 pb-3">
                              {total === 0 && (
                                <p className="text-xs text-muted-foreground">
                                  нет данных - добавьте навыки выше
                                </p>
                              )}
                              {matched.length > 0 && (
                                <div>
                                  <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                                    Совпало ({matched.length})
                                  </div>
                                  <div className="flex flex-wrap gap-1.5">
                                    {matched.map((s, i) => (
                                      <Badge
                                        key={i}
                                        tone="success"
                                        className="max-w-full text-xs"
                                        title="Есть у вас"
                                      >
                                        <CheckCircle2 className="size-3 shrink-0" />
                                        <span className="break-words">{s.text}</span>
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {missing.length > 0 && (
                                <div>
                                  <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                                    Не хватает ({missing.length})
                                  </div>
                                  <div className="flex flex-wrap gap-1.5">
                                    {missing.map((s, i) => (
                                      <Badge
                                        key={i}
                                        tone="neutral"
                                        className="max-w-full text-xs"
                                        title="Нет у вас - нажмите +, чтобы добавить"
                                      >
                                        <span className="break-words">{s.text}</span>
                                        <button
                                          onClick={() => void handleAddSkill(s.text)}
                                          disabled={saving}
                                          className="cursor-pointer rounded p-0.5 text-muted-foreground transition-colors duration-200 hover:bg-muted hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                                          aria-label={`Добавить «${s.text}»`}
                                        >
                                          <Plus className="size-3" />
                                        </button>
                                      </Badge>
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
                  variant="secondary"
                  onClick={() => onNavigate("data")}
                  className="cursor-pointer"
                  iconLeft={<Database className="size-4" />}
                >
                  Открыть результаты
                </Button>
              </div>
            </>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
