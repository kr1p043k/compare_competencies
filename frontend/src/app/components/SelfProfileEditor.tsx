import { useEffect, useState } from "react";
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
import { Loader2, Plus, User, X } from "lucide-react";
import { apiFetch } from "../../lib/auth";

interface SelfData {
  profile: string;
  target_level: string;
  skills: string[];
  user_added: string[];
}

const LEVELS = [
  { value: "junior", label: "Junior" },
  { value: "middle", label: "Middle" },
  { value: "senior", label: "Senior" },
];

/** Свой профиль студента: уровень + свои навыки (удаление — только своих). */
export function SelfProfileEditor() {
  const [data, setData] = useState<SelfData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newSkill, setNewSkill] = useState("");

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await apiFetch("/api/profiles/self");
      if (!r.ok) throw new Error("Не удалось загрузить профиль");
      setData(await r.json());
    } catch (e: any) {
      setError(e?.message || "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const patch = async (body: Record<string, unknown>) => {
    setSaving(true);
    setError(null);
    try {
      const r = await apiFetch("/api/profiles/self", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || "Не удалось сохранить");
      }
      setData(await r.json());
    } catch (e: any) {
      setError(e?.message || "Ошибка сохранения");
    } finally {
      setSaving(false);
    }
  };

  const owned = new Set((data?.user_added || []).map((s) => s.toLowerCase()));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <User className="size-5 text-emerald-600" />
          Мой профиль
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && (
          <p className="text-sm text-gray-500 dark:text-slate-400">Загрузка…</p>
        )}
        {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        {data && (
          <>
            <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
              <span className="text-sm font-medium text-gray-700 dark:text-slate-300">Уровень</span>
              <Select
                value={data.target_level}
                onValueChange={(v) => patch({ target_level: v })}
                disabled={saving}
              >
                <SelectTrigger className="h-10 w-full sm:w-44">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LEVELS.map((l) => (
                    <SelectItem key={l.value} value={l.value}>
                      {l.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <div className="text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                Мои навыки ({data.skills.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.skills.map((s) => {
                  const mine = owned.has(s.toLowerCase());
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
                          onClick={() => patch({ remove_skills: [s] })}
                          disabled={saving}
                          className="ml-1 hover:text-red-300 cursor-pointer"
                          aria-label={`Удалить ${s}`}
                        >
                          <X className="size-3" />
                        </button>
                      )}
                    </Badge>
                  );
                })}
                {data.skills.length === 0 && (
                  <span className="text-sm text-gray-400 dark:text-slate-500">
                    Пока пусто — добавьте свои навыки ниже.
                  </span>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <Input
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newSkill.trim()) {
                    patch({ add_skills: [newSkill.trim()] });
                    setNewSkill("");
                  }
                }}
                placeholder="Новый навык, Enter — добавить"
                className="h-10"
              />
              <Button
                onClick={() => {
                  if (newSkill.trim()) {
                    patch({ add_skills: [newSkill.trim()] });
                    setNewSkill("");
                  }
                }}
                disabled={saving || !newSkill.trim()}
                className="h-10 bg-emerald-700 hover:bg-emerald-800 text-white"
              >
                {saving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default SelfProfileEditor;
