import { useEffect, useState } from "react";
import { Badge, Button, Card, CardBody, CardHeader, inputClass } from "../../design";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Loader2, Plus, User, X } from "lucide-react";
import { request } from "../../lib/api";

interface SelfData {
  profile: string;
  target_level: string;
  skills: string[];
  user_added: string[];
  competencies?: string[];
}

const LEVELS = [
  { value: "junior", label: "Junior" },
  { value: "middle", label: "Middle" },
  { value: "senior", label: "Senior" },
];

/** Свой профиль: ФИО, уровень, компетенции с навыками, свои навыки. */
export function SelfProfileEditor({ displayName, email }: { displayName?: string; email?: string }) {
  const [data, setData] = useState<SelfData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newSkill, setNewSkill] = useState("");

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await request<SelfData>("/api/profiles/self"));
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
      setData(await request<SelfData>("/api/profiles/self", { method: "PATCH", body }));
    } catch (e: any) {
      setError(e?.message || "Ошибка сохранения");
    } finally {
      setSaving(false);
    }
  };

  const owned = new Set((data?.user_added || []).map((s) => s.toLowerCase()));

  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <User className="size-5 text-success" />
            Мой профиль
          </span>
        }
        description={
          displayName || email
            ? `${displayName || "-"}${email && displayName !== email ? ` · ${email}` : ""}`
            : undefined
        }
      />
      <CardBody className="space-y-4">
        {loading && (
          <p className="text-sm text-muted-foreground">Загрузка…</p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        {data && (
          <>
            <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
              <span className="text-sm font-medium text-foreground">Уровень</span>
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

            {(data.competencies || []).length > 0 && (
              <div>
                <div className="text-sm font-medium text-foreground mb-2">
                  Текущие компетенции ({(data.competencies || []).length})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(data.competencies || []).map((c) => (
                    <Badge key={c} tone="neutral" className="text-xs">
                      {c}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="text-sm font-medium text-foreground mb-2">
                Мои навыки ({data.skills.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {data.skills.map((s) => {
                  const mine = owned.has(s.toLowerCase());
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
                          onClick={() => patch({ remove_skills: [s] })}
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
                {data.skills.length === 0 && (
                  <span className="text-sm text-subtle-foreground">
                    Пока пусто - добавьте свои навыки ниже.
                  </span>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <input
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newSkill.trim()) {
                    patch({ add_skills: [newSkill.trim()] });
                    setNewSkill("");
                  }
                }}
                placeholder="Новый навык, Enter - добавить"
                aria-label="Новый навык"
                className={inputClass}
              />
              <Button
                variant="success"
                onClick={() => {
                  if (newSkill.trim()) {
                    patch({ add_skills: [newSkill.trim()] });
                    setNewSkill("");
                  }
                }}
                disabled={saving || !newSkill.trim()}
                className="shrink-0 cursor-pointer"
                aria-label="Добавить навык"
              >
                {saving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}

export default SelfProfileEditor;
