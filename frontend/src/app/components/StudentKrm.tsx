import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { BookOpen, CheckCircle2, ChevronDown, GraduationCap, Loader2, User } from "lucide-react";
import { apiFetch } from "../../lib/auth";

interface KrmData {
  direction: string;
  direction_name: string;
  profile: string;
  disciplines: Array<{
    discipline: string;
    competencies: Array<{
      code: string;
      skills: Array<{ text: string; student_has: boolean }>;
    }>;
  }>;
  merged: Array<{ skill: string; source: string; code: string; discipline: string }>;
  counts: { krm: number; student: number; overlap: number; merged: number; student_only: number };
}

export function StudentKrm() {
  const [dirs, setDirs] = useState<Array<{ dir_code: string; name: string }>>([]);
  const [profiles, setProfiles] = useState<string[]>(["base", "dc", "top_dc"]);
  const [dir, setDir] = useState("");
  const [prof, setProf] = useState("base");
  const [data, setData] = useState<KrmData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openDisc, setOpenDisc] = useState<Record<string, boolean>>({});

  useEffect(() => {
    apiFetch("/api/krm/directions")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.directions?.length) {
          setDirs(d.directions);
          setDir(d.directions[0].dir_code);
        }
      })
      .catch(() => {});
    apiFetch("/api/profiles")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (Array.isArray(d?.profiles) && d.profiles.length) setProfiles(d.profiles);
      })
      .catch(() => {});
  }, []);

  const load = async () => {
    if (!dir || !prof) return;
    setLoading(true);
    setError(null);
    try {
      const r = await apiFetch(
        `/api/krm/student/competencies?direction=${encodeURIComponent(dir)}&profile=${encodeURIComponent(prof)}`
      );
      if (!r.ok) throw new Error("Не удалось загрузить компетенции");
      setData(await r.json());
    } catch (e: any) {
      setError(e?.message || "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <GraduationCap className="size-5 text-indigo-600" />
          Мои компетенции: программа и я
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col sm:flex-row gap-2">
          <Select value={dir} onValueChange={setDir}>
            <SelectTrigger className="h-10 flex-1">
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
            <SelectTrigger className="h-10 w-full sm:w-44">
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
          <Button onClick={load} disabled={loading || !dir} className="h-10 bg-blue-700 hover:bg-blue-800 text-white">
            {loading ? <Loader2 className="size-4 animate-spin" /> : <BookOpen className="size-4 mr-2" />}
            Показать
          </Button>
        </div>

        {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

        <AnimatePresence>
          {data && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                {[
                  ["КРМ", data.counts.krm],
                  ["Мои", data.counts.student],
                  ["Совпало", data.counts.overlap],
                  ["Только мои", data.counts.student_only],
                ].map(([label, v]) => (
                  <div key={label as string} className="rounded-lg bg-slate-50 dark:bg-slate-900 p-2.5">
                    <div className="text-xl font-bold text-slate-900 dark:text-slate-100">{v}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
                  </div>
                ))}
              </div>

              <div className="space-y-2">
                {data.disciplines.map((d) => {
                  const total = d.competencies.reduce((a, c) => a + c.skills.length, 0);
                  const has = d.competencies.reduce(
                    (a, c) => a + c.skills.filter((s) => s.student_has).length, 0
                  );
                  const open = !!openDisc[d.discipline];
                  return (
                    <div key={d.discipline} className="border border-slate-200 dark:border-slate-700 rounded-lg">
                      <button
                        onClick={() => setOpenDisc((p) => ({ ...p, [d.discipline]: !p[d.discipline] }))}
                        className="w-full flex items-center justify-between px-3 py-2.5 text-left"
                      >
                        <span className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">
                          {d.discipline}
                        </span>
                        <span className="flex items-center gap-2 shrink-0">
                          <Badge variant={has > 0 ? "default" : "secondary"} className="text-[11px]">
                            {has}/{total}
                          </Badge>
                          <ChevronDown className={`size-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`} />
                        </span>
                      </button>
                      {open && (
                        <div className="px-3 pb-3 space-y-2">
                          {d.competencies.map((c) => (
                            <div key={c.code}>
                              <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">{c.code}</div>
                              <ul className="space-y-1">
                                {c.skills.map((s, i) => (
                                  <li key={i} className="flex items-start gap-1.5 text-xs text-slate-700 dark:text-slate-300">
                                    {s.student_has ? (
                                      <CheckCircle2 className="size-3.5 mt-0.5 text-emerald-600 shrink-0" />
                                    ) : (
                                      <User className="size-3.5 mt-0.5 text-slate-300 dark:text-slate-600 shrink-0" />
                                    )}
                                    <span>{s.text}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}

export default StudentKrm;
