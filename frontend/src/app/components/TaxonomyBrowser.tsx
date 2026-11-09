import { useEffect, useMemo, useState } from "react";
import { BookOpen, ChevronDown, Search, Loader2 } from "lucide-react";
import { api } from "../api";
import { reportError } from "./../../lib/errors";
import { Badge, Button, Card, EmptyState, PageHeader, inputClass } from "../../design";
import { cn } from "./ui/utils";

interface TaxCategory {
  id: string;
  label: string;
  icon: string;
  total: number;
  skills: string[];
}

/** Тон бейджа статуса предложения навыка. */
function suggestStatusTone(status: string): "success" | "danger" | "warning" {
  if (status === "approved") return "success";
  if (status === "rejected") return "danger";
  return "warning";
}

/** Просмотр таксономии навыков: категории, поиск, состав. Только чтение. */
export function TaxonomyBrowser({ showSuggest = false }: { showSuggest?: boolean }) {
  const [cats, setCats] = useState<TaxCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [skillInput, setSkillInput] = useState("");
  const [catHint, setCatHint] = useState("");
  const [suggestMsg, setSuggestMsg] = useState("");
  const [suggestBusy, setSuggestBusy] = useState(false);
  const [myList, setMyList] = useState<any[]>([]);

  const loadMine = () => {
    if (!showSuggest) return;
    api<{ suggestions: unknown[] }>("/teacher/skills/suggestions").then((d) => setMyList(d?.suggestions || [])).catch((err) => reportError("app/components/TaxonomyBrowser.tsx", err));
  };

  const submitSuggest = async () => {
    const skill = skillInput.trim();
    if (!skill || suggestBusy) return;
    setSuggestBusy(true);
    setSuggestMsg("");
    try {
      await api("/teacher/skills/suggest", {
        method: "POST",
        body: JSON.stringify({ skill, category_hint: catHint }),
      });
      setSkillInput("");
      setCatHint("");
      setSuggestMsg("Отправлено на модерацию админу");
      loadMine();
    } catch (e: any) {
      setSuggestMsg(e?.message || "Ошибка отправки");
    } finally {
      setSuggestBusy(false);
    }
  };

  useEffect(() => {
    let alive = true;
    api<{ categories: TaxCategory[] }>("/taxonomy/categories")
      .then((d) => {
        if (alive) setCats(d?.categories || []);
      })
      .catch((e) => {
        if (alive) setError(e?.message || "Ошибка загрузки");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    loadMine();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showSuggest]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cats;
    return cats
      .map((c) => ({
        ...c,
        skills: c.skills.filter((s) => s.toLowerCase().includes(q)),
      }))
      .filter((c) => c.skills.length > 0 || c.label.toLowerCase().includes(q));
  }, [cats, query]);

  const totalSkills = useMemo(
    () => cats.reduce((n, c) => n + (c.total || 0), 0),
    [cats]
  );

  return (
    <div className="space-y-6">
      <PageHeader
        icon={BookOpen}
        title="Таксономия навыков"
        description="Категории навыков и предложения по их добавлению."
      />
      <Card className="overflow-hidden">

      <div className="flex items-center gap-3 px-6 py-3 border-b border-border bg-muted">
          <div className="flex items-center justify-center w-9 h-9 bg-primary text-primary-foreground rounded-lg shrink-0">
            <BookOpen className="size-5" />
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="m-0 leading-none text-lg font-semibold text-foreground">
              Таксономия навыков
            </h4>
            <p className="text-sm text-muted-foreground">
              {cats.length} категорий · {totalSkills} навыков · только просмотр
            </p>
          </div>
        </div>
      {showSuggest && (
        <div className="border-b border-border bg-muted/60 px-6 py-4 space-y-3">
          <p className="text-sm font-medium text-foreground">
            Предложить навык в таксономию
          </p>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              value={skillInput}
              onChange={(e) => setSkillInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitSuggest();
              }}
              placeholder="Название навыка..."
              maxLength={80}
              className={cn(inputClass, "flex-1 h-9")}
            />
            <select
              value={catHint}
              onChange={(e) => setCatHint(e.target.value)}
              className={cn(inputClass, "h-9 w-auto px-2 sm:max-w-48")}
            >
              <option value="">Категория (необязательно)</option>
              {cats.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <Button
              onClick={submitSuggest}
              disabled={suggestBusy || !skillInput.trim()}
              className="h-9"
            >
              {suggestBusy ? "..." : "Отправить"}
            </Button>
          </div>
          {suggestMsg && (
            <p className="text-xs text-muted-foreground">{suggestMsg}</p>
          )}
          {myList.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {myList.map((s: any) => (
                <Badge
                  key={s.id}
                  tone={suggestStatusTone(s.status)}
                  className="text-xs"
                  title={`${s.created_at || ""}${s.category_hint ? ` · ${s.category_hint}` : ""}`}
                >
                  {s.skill} · {s.status === "approved" ? "принят" : s.status === "rejected" ? "отклонён" : "на модерации"}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="p-6 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Найти навык или категорию..."
            className={cn(inputClass, "pl-10")}
          />
        </div>
        {loading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground">
            <Loader2 className="size-6 animate-spin" />
          </div>
        ) : error ? (
          <p className="text-sm text-destructive text-center py-6">{error}</p>
        ) : filtered.length === 0 ? (
          <EmptyState icon={<Search className="size-8" />} title="Ничего не найдено" description="Попробуйте другой запрос." />
        ) : (
          <div className="divide-y divide-border border border-border rounded-lg overflow-hidden">
            {filtered.map((c) => {
              const isOpen = open[c.id] ?? query.trim().length > 0;
              return (
                <div key={c.id}>
                  <button
                    onClick={() => setOpen((p) => ({ ...p, [c.id]: !(p[c.id] ?? query.trim().length > 0) }))}
                    className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-muted/40 cursor-pointer"
                  >
                    <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", isOpen && "rotate-180")} />
                    <span className="mr-1">{c.icon}</span>
                    <span className="text-sm font-medium text-foreground">{c.label}</span>
                    <span className="ml-auto text-xs text-subtle-foreground tabular-nums">
                      {c.skills.length}{c.skills.length !== c.total ? ` из ${c.total}` : ""}
                    </span>
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-3 pt-1 flex flex-wrap gap-1.5">
                      {c.skills.map((s) => (
                        <Badge key={s} tone="neutral" className="text-xs">
                          {s}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
    </div>
  );
}
