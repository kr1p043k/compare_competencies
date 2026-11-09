// NOTE: student-only block - renders inside StudentDashboard for students only.
// All writes go to the viewer's own profile via PATCH /api/profiles/self
// (add_skills). No taxonomy endpoint is called here and no taxonomy
// category names are rendered - only KRM program data (disciplines,
// competency codes, skill texts) from /api/krm/student/competencies.
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  GraduationCap,
  Loader2,
  Plus,
  Search,
  X,
} from "lucide-react";
import { reportError } from "./../../lib/errors";
import { request } from "../../lib/api";

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
  merged: Array<{ skill: string; source: string; code: string; discipline: string }>;
  counts: { krm: number; student: number; overlap: number; merged: number; student_only: number };
}

interface MissingSkill {
  text: string;
  code: string;
  discipline: string;
}

type MatchFilter = "all" | "matched" | "missing";

const norm = (s: string) => s.toLowerCase().trim();

/** Client-side approximation of the server matcher - optimistic UI only. */
function looselyMatches(a: string, b: string): boolean {
  const x = norm(a);
  const y = norm(b);
  if (!x || !y) return false;
  if (x === y) return true;
  return x.length > 3 && y.length > 3 && (x.includes(y) || y.includes(x));
}

/** Optimistically mark KRM items covered by `text`; server reload reconciles. */
function applyOptimisticAdd(prev: KrmData, text: string): KrmData {
  let newlyMatched = 0;
  const disciplines = prev.disciplines.map((d) => ({
    ...d,
    competencies: d.competencies.map((c) => ({
      ...c,
      skills: c.skills.map((s) => {
        if (!s.student_has && looselyMatches(s.text, text)) {
          newlyMatched += 1;
          return { ...s, student_has: true };
        }
        return s;
      }),
    })),
  }));
  return {
    ...prev,
    disciplines,
    counts: {
      ...prev.counts,
      student: prev.counts.student + 1,
      overlap: prev.counts.overlap + (newlyMatched > 0 ? 1 : 0),
      student_only: prev.counts.student_only + (newlyMatched > 0 ? 0 : 1),
    },
  };
}

async function patchSelf(body: Record<string, unknown>): Promise<void> {
  await request<void>("/api/profiles/self", { method: "PATCH", body });
}

export function StudentKrm({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [dirs, setDirs] = useState<Array<{ dir_code: string; name: string }>>([]);
  const [profiles, setProfiles] = useState<string[]>(["base", "dc", "top_dc"]);
  const [dir, setDir] = useState("");
  const [prof, setProf] = useState("base");
  const [data, setData] = useState<KrmData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openDisc, setOpenDisc] = useState<Record<string, boolean>>({});
  const [openCodes, setOpenCodes] = useState<Record<string, boolean>>({});
  const [discQuery, setDiscQuery] = useState("");
  const [matchFilter, setMatchFilter] = useState<MatchFilter>("all");
  const [newSkill, setNewSkill] = useState("");
  const [compPick, setCompPick] = useState("");
  const [savingSkill, setSavingSkill] = useState(false);
  const [savingComp, setSavingComp] = useState(false);
  const [addMsg, setAddMsg] = useState<string | null>(null);
  const [addErr, setAddErr] = useState<string | null>(null);
  const [mineAdded, setMineAdded] = useState<string[]>([]);

  useEffect(() => {
    request<{ directions?: Array<{ dir_code: string; name: string }> }>("/api/krm/directions")
      .then((d) => {
        if (d?.directions?.length) {
          setDirs(d.directions);
          setDir(d.directions[0].dir_code);
        }
      })
      .catch((err) => reportError("app/components/StudentKrm.tsx", err));
    request<{ profiles?: string[] }>("/api/profiles")
      .then((d) => {
        if (Array.isArray(d?.profiles) && d.profiles.length) setProfiles(d.profiles);
      })
      .catch((err) => reportError("app/components/StudentKrm.tsx", err));
  }, []);

  const load = async () => {
    if (!dir || !prof) return;
    setLoading(true);
    setError(null);
    try {
      setData(
        await request<KrmData>(
          `/api/krm/student/competencies?direction=${encodeURIComponent(dir)}&profile=${encodeURIComponent(prof)}`,
        ),
      );
    } catch (e: any) {
      setError(e?.message || "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  };

  const discStats = useMemo(() => {
    if (!data) return [];
    return data.disciplines.map((d) => {
      const total = d.competencies.reduce((a, c) => a + c.skills.length, 0);
      const has = d.competencies.reduce(
        (a, c) => a + c.skills.filter((s) => s.student_has).length,
        0
      );
      return { name: d.discipline, total, has };
    });
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = norm(discQuery);
    return data.disciplines.filter((d) => {
      if (q && !norm(d.discipline).includes(q)) return false;
      if (matchFilter === "all") return true;
      const total = d.competencies.reduce((a, c) => a + c.skills.length, 0);
      const has = d.competencies.reduce(
        (a, c) => a + c.skills.filter((s) => s.student_has).length,
        0
      );
      if (total === 0) return matchFilter === "missing";
      return matchFilter === "matched" ? has > 0 : has < total;
    });
  }, [data, discQuery, matchFilter]);

  /** Program base items not yet owned - source for the competency picker. */
  const missingSkills = useMemo<MissingSkill[]>(() => {
    if (!data) return [];
    const out: MissingSkill[] = [];
    for (const d of data.disciplines) {
      for (const c of d.competencies) {
        for (const s of c.skills) {
          if (!s.student_has) out.push({ text: s.text, code: c.code, discipline: d.discipline });
        }
      }
    }
    return out;
  }, [data]);

  const rememberMine = (text: string) =>
    setMineAdded((prev) =>
      prev.some((s) => norm(s) === norm(text)) ? prev : [...prev, text]
    );

  const handleAddSkill = async () => {
    const text = newSkill.trim();
    if (!text || savingSkill) return;
    const backup = data;
    setSavingSkill(true);
    setAddErr(null);
    setAddMsg(null);
    if (backup) setData(applyOptimisticAdd(backup, text));
    setNewSkill("");
    try {
      await patchSelf({ add_skills: [text] });
      rememberMine(text);
      setAddMsg(`«${text}» - добавлено в ваши навыки`);
      await load();
    } catch (e: any) {
      if (backup) setData(backup);
      setNewSkill(text);
      setAddErr(e?.message || "Не удалось добавить навык");
    } finally {
      setSavingSkill(false);
    }
  };

  const handleAddCompetency = async (raw?: string) => {
    const text = (raw ?? compPick).trim();
    if (!text || savingComp) return;
    const backup = data;
    setSavingComp(true);
    setAddErr(null);
    setAddMsg(null);
    if (backup) setData(applyOptimisticAdd(backup, text));
    try {
      await patchSelf({ add_skills: [text] });
      rememberMine(text);
      setCompPick("");
      setAddMsg(`«${text}» - добавлено в ваши навыки`);
      await load();
    } catch (e: any) {
      if (backup) setData(backup);
      setAddErr(e?.message || "Не удалось добавить компетенцию");
    } finally {
      setSavingComp(false);
    }
  };

  const setAllDisc = (open: boolean) => {
    if (!data) return;
    const next: Record<string, boolean> = {};
    for (const d of data.disciplines) next[d.discipline] = open;
    setOpenDisc(next);
  };

  const stats = data
    ? [
        { label: "Программа", value: data.direction_name || data.direction, long: true },
        { label: "Base", value: data.profile, long: true },
        { label: "КРМ", value: String(data.counts.krm) },
        { label: "Мои", value: String(data.counts.student) },
        { label: "Совпало", value: String(data.counts.overlap) },
        { label: "Только мои", value: String(data.counts.student_only) },
      ]
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <GraduationCap className="size-5 text-primary" />
          Мои компетенции: программа и я
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Сравнение программы с вашими навыками - компактный вид по дисциплинам
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-2">
          <Select value={dir} onValueChange={setDir}>
            <SelectTrigger className="h-10 flex-1" aria-label="Направление">
              <SelectValue placeholder="Направление…" />
            </SelectTrigger>
            <SelectContent>
              {dirs.map((d) => (
                <SelectItem key={d.dir_code} value={d.dir_code}>
                  {d.dir_code} · {d.name || "Программа"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={prof} onValueChange={setProf}>
            <SelectTrigger className="h-10 w-full sm:w-44" aria-label="Профиль">
              <SelectValue placeholder="Профиль…" />
            </SelectTrigger>
            <SelectContent>
              {profiles.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={load} disabled={loading || !dir} className="h-10 bg-primary hover:bg-primary/90 text-primary-foreground cursor-pointer">
            {loading ? <Loader2 className="size-4 animate-spin" /> : <BookOpen className="size-4 mr-2" />}
            Показать
          </Button>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}

        <AnimatePresence>
          {data && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                {stats.map((s) => (
                  <div
                    key={s.label}
                    title={s.value}
                    className="rounded-lg border border-border bg-card p-2.5 text-center transition-colors duration-200  dark:bg-card"
                  >
                    <div className="truncate text-lg font-bold text-foreground">
                      {s.value}
                    </div>
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>

              <div className="sticky top-0 z-10 bg-card/95 py-2 backdrop-blur transition-colors duration-200">
                <div className="flex flex-col sm:flex-row gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground" />
                    <Input
                      value={discQuery}
                      onChange={(e) => setDiscQuery(e.target.value)}
                      placeholder="Фильтр дисциплин…"
                      aria-label="Фильтр дисциплин"
                      className="h-9 pl-8 pr-8"
                    />
                    {discQuery && (
                      <button
                        onClick={() => setDiscQuery("")}
                        className="absolute right-2 top-1/2 -translate-y-1/2 cursor-pointer rounded p-0.5 text-subtle-foreground transition-colors duration-200 hover:text-foreground dark:hover:text-foreground"
                        aria-label="Очистить фильтр"
                      >
                        <X className="size-4" />
                      </button>
                    )}
                  </div>
                  <Select value={matchFilter} onValueChange={(v) => setMatchFilter(v as MatchFilter)}>
                    <SelectTrigger className="h-9 w-full sm:w-44" aria-label="Фильтр совпадений">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Все дисциплины</SelectItem>
                      <SelectItem value="matched">Есть совпадения</SelectItem>
                      <SelectItem value="missing">Без совпадений</SelectItem>
                    </SelectContent>
                  </Select>
                  <div className="flex gap-1.5">
                    <Button variant="outline" size="sm" onClick={() => setAllDisc(true)} className="h-9 cursor-pointer">
                      Развернуть
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setAllDisc(false)} className="h-9 cursor-pointer">
                      Свернуть
                    </Button>
                  </div>
                </div>
                <p className="mt-1.5 text-xs text-muted-foreground">
                  Показано {filtered.length} из {data.disciplines.length} дисциплин
                </p>
              </div>

              {data.disciplines.length === 0 && (
                <p className="rounded-lg border border-dashed border-border-strong px-3 py-4 text-center text-sm text-muted-foreground  dark:text-subtle-foreground">
                  нет данных - добавьте навыки ниже
                </p>
              )}

              {data.disciplines.length > 0 && filtered.length === 0 && (
                <p className="rounded-lg border border-dashed border-border-strong px-3 py-4 text-center text-sm text-muted-foreground  dark:text-subtle-foreground">
                  По фильтру ничего не найдено - измените запрос или сбросьте фильтр
                </p>
              )}

              <div className="space-y-2">
                {filtered.map((d) => {
                  const st = discStats.find((s) => s.name === d.discipline);
                  const total = st?.total ?? 0;
                  const has = st?.has ?? 0;
                  const open = !!openDisc[d.discipline];
                  return (
                    <div key={d.discipline} className="rounded-lg border border-border bg-card transition-colors duration-200  dark:bg-card">
                      <button
                        onClick={() => setOpenDisc((p) => ({ ...p, [d.discipline]: !p[d.discipline] }))}
                        className="flex w-full cursor-pointer items-center justify-between gap-2 px-3 py-2.5 text-left transition-colors duration-200 hover:bg-muted dark:hover:bg-muted/50"
                        aria-expanded={open}
                      >
                        <span className="truncate text-sm font-medium text-foreground">
                          {d.discipline}
                        </span>
                        <span className="flex shrink-0 items-center gap-2">
                          <Badge variant={has > 0 ? "default" : "secondary"} className="text-xs">
                            {has}/{total}
                          </Badge>
                          <ChevronDown className={`size-4 text-subtle-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                        </span>
                      </button>
                      {open && (
                        <div className="space-y-2 px-3 pb-3">
                          {total === 0 && (
                            <p className="text-xs text-muted-foreground">
                              нет данных - добавьте навыки ниже
                            </p>
                          )}
                          {d.competencies.map((c) => {
                            const key = `${d.discipline}::${c.code}`;
                            const codeOpen = openCodes[key] ?? true;
                            const cHas = c.skills.filter((s) => s.student_has).length;
                            return (
                              <div key={c.code} className="rounded-md bg-muted transition-colors duration-200">
                                <button
                                  onClick={() => setOpenCodes((p) => ({ ...p, [key]: !codeOpen }))}
                                  className="flex w-full cursor-pointer items-center justify-between gap-2 px-2.5 py-1.5 text-left transition-colors duration-200 hover:bg-muted dark:hover:bg-muted"
                                  aria-expanded={codeOpen}
                                >
                                  <span className="text-xs font-semibold text-muted-foreground">
                                    {c.code}
                                  </span>
                                  <span className="flex shrink-0 items-center gap-1.5">
                                    <span className="text-xs text-subtle-foreground">
                                      {cHas}/{c.skills.length}
                                    </span>
                                    <ChevronDown className={`size-3.5 text-subtle-foreground transition-transform duration-200 ${codeOpen ? "rotate-180" : ""}`} />
                                  </span>
                                </button>
                                {codeOpen && (
                                  <div className="flex flex-wrap gap-1.5 px-2.5 pb-2.5">
                                    {c.skills.map((s, i) =>
                                      s.student_has ? (
                                        <span
                                          key={i}
                                          title="Есть у вас"
                                          className="inline-flex max-w-full items-center gap-1 rounded-md border border-success/30 bg-success-soft px-2 py-0.5 text-xs font-medium text-success transition-colors duration-200 dark:border-success/40 dark:bg-success/10 dark:text-success"
                                        >
                                          <CheckCircle2 className="size-3 shrink-0" />
                                          <span className="break-words">{s.text}</span>
                                        </span>
                                      ) : (
                                        <span
                                          key={i}
                                          title="Нет у вас - нажмите +, чтобы добавить"
                                          className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-card px-2 py-0.5 text-xs text-foreground transition-colors duration-200  dark:bg-card"
                                        >
                                          <span className="break-words">{s.text}</span>
                                          <button
                                            onClick={() => void handleAddCompetency(s.text)}
                                            disabled={savingComp}
                                            className="cursor-pointer rounded p-0.5 text-subtle-foreground transition-colors duration-200 hover:bg-muted hover:text-success focus-visible:outline-2 dark:hover:bg-muted dark:hover:text-success"
                                            aria-label={`Добавить «${s.text}»`}
                                          >
                                            <Plus className="size-3" />
                                          </button>
                                        </span>
                                      )
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {onNavigate && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 ">
            <p className="text-xs text-muted-foreground">
              Технологии и свои профили живут в профиле
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onNavigate("profile")}
              className="cursor-pointer transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              Открыть мой профиль
              <ArrowRight className="ml-1.5 size-3.5" />
            </Button>
          </div>
        )}
        <div className="grid gap-3 border-t border-border pt-4 sm:grid-cols-2 ">
          <div className="rounded-lg border border-border p-3 ">
            <div className="text-sm font-medium text-foreground">
              Добавить свой навык
            </div>
            <p className="text-xs text-muted-foreground">
              Сохраняется в ваши навыки
            </p>
            <div className="mt-2 flex gap-2">
              <Input
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newSkill.trim()) void handleAddSkill();
                }}
                placeholder="Новый навык, Enter - добавить"
                aria-label="Новый навык"
                className="h-9"
              />
              <Button
                onClick={() => void handleAddSkill()}
                disabled={savingSkill || !newSkill.trim()}
                className="h-9 shrink-0 bg-success text-white hover:bg-success/90 cursor-pointer"
                aria-label="Добавить навык"
              >
                {savingSkill ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
          </div>
          <div className="rounded-lg border border-border p-3 ">
            <div className="text-sm font-medium text-foreground">
              Добавить из программы
            </div>
            <p className="text-xs text-muted-foreground">
              {data
                ? missingSkills.length > 0
                  ? `Не хватает в ваших навыках: ${missingSkills.length}`
                  : "Всё из программы уже у вас"
                : "Сначала нажмите «Показать», чтобы загрузить программу"}
            </p>
            <div className="mt-2 flex gap-2">
              <Select value={compPick} onValueChange={setCompPick} disabled={!data || missingSkills.length === 0}>
                <SelectTrigger className="h-9 flex-1" aria-label="Компетенция из программы">
                  <SelectValue placeholder="Выберите из программы…" />
                </SelectTrigger>
                <SelectContent>
                  {missingSkills.map((m, i) => (
                    <SelectItem key={`${m.discipline}::${m.code}::${i}`} value={m.text}>
                      {m.text} · {m.code}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={() => void handleAddCompetency()}
                disabled={savingComp || !compPick.trim()}
                className="h-9 shrink-0 bg-success text-white hover:bg-success/90 cursor-pointer"
                aria-label="Добавить компетенцию"
              >
                {savingComp ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
          </div>
        </div>

        {addErr && <p className="text-sm text-destructive">{addErr}</p>}
        {addMsg && <p className="text-sm text-success">{addMsg}</p>}
        {mineAdded.length > 0 && (
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted-foreground">
              Добавлено вами в этой сессии ({mineAdded.length})
            </div>
            <div className="flex flex-wrap gap-1.5">
              {mineAdded.map((s) => (
                <span
                  key={s}
                  className="inline-flex max-w-full items-center gap-1 rounded-md border border-dashed border-success/40 bg-success-soft px-2 py-0.5 text-xs text-success transition-colors duration-200 dark:border-success/40 dark:bg-success/10 dark:text-success"
                >
                  <CheckCircle2 className="size-3 shrink-0" />
                  <span className="break-words">{s}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default StudentKrm;
