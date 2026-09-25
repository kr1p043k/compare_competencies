import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { BookOpen, ChevronDown, Search, Loader2 } from "lucide-react";
import { api } from "../api";

interface TaxCategory {
  id: string;
  label: string;
  icon: string;
  total: number;
  skills: string[];
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
    api("/teacher/skills/suggestions").then((d) => setMyList(d?.suggestions || [])).catch(() => {});
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
    api("/taxonomy/categories")
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
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
      <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 bg-violet-600 rounded-lg shrink-0">
            <BookOpen className="size-5 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg font-semibold text-gray-900 dark:text-slate-100">
              Таксономия навыков
            </CardTitle>
            <CardDescription className="text-sm text-gray-600 dark:text-slate-400">
              {cats.length} категорий · {totalSkills} навыков · только просмотр
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      {showSuggest && (
        <div className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900/60 px-6 py-4 space-y-3">
          <p className="text-sm font-medium text-gray-800 dark:text-slate-200">
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
              className="flex-1 h-9 px-3 text-sm rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 outline-none"
            />
            <select
              value={catHint}
              onChange={(e) => setCatHint(e.target.value)}
              className="h-9 px-2 text-sm rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 outline-none sm:max-w-48"
            >
              <option value="">Категория (необязательно)</option>
              {cats.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <button
              onClick={submitSuggest}
              disabled={suggestBusy || !skillInput.trim()}
              className="h-9 px-4 text-sm font-medium text-white bg-violet-600 hover:bg-violet-700 rounded-lg transition-colors cursor-pointer disabled:opacity-50 whitespace-nowrap"
            >
              {suggestBusy ? "..." : "Отправить"}
            </button>
          </div>
          {suggestMsg && (
            <p className="text-xs text-gray-500 dark:text-slate-400">{suggestMsg}</p>
          )}
          {myList.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {myList.map((s: any) => (
                <span
                  key={s.id}
                  title={`${s.created_at || ""}${s.category_hint ? ` · ${s.category_hint}` : ""}`}
                  className={`px-2 py-0.5 text-xs rounded-full border ${
                    s.status === "approved"
                      ? "bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700"
                      : s.status === "rejected"
                        ? "bg-red-100 dark:bg-red-950/40 text-red-700 dark:text-red-300 border-red-300 dark:border-red-700"
                        : "bg-amber-100 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-700"
                  }`}
                >
                  {s.skill} · {s.status === "approved" ? "принят" : s.status === "rejected" ? "отклонён" : "на модерации"}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      <CardContent className="p-6 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-slate-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Найти навык или категорию..."
            className="w-full h-10 pl-10 pr-3 text-sm rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 outline-none"
          />
        </div>
        {loading ? (
          <div className="flex items-center justify-center py-10 text-gray-400">
            <Loader2 className="size-6 animate-spin" />
          </div>
        ) : error ? (
          <p className="text-sm text-red-600 text-center py-6">{error}</p>
        ) : filtered.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-slate-400 text-center py-6">
            Ничего не найдено
          </p>
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg overflow-hidden">
            {filtered.map((c) => {
              const isOpen = open[c.id] ?? query.trim().length > 0;
              return (
                <div key={c.id}>
                  <button
                    onClick={() => setOpen((p) => ({ ...p, [c.id]: !(p[c.id] ?? query.trim().length > 0) }))}
                    className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-gray-50 dark:hover:bg-slate-800/60 cursor-pointer"
                  >
                    <ChevronDown className={`size-4 text-gray-400 transition-transform ${isOpen ? "rotate-180" : ""}`} />
                    <span className="mr-1">{c.icon}</span>
                    <span className="text-sm font-medium text-gray-800 dark:text-slate-200">{c.label}</span>
                    <span className="ml-auto text-xs text-gray-400 dark:text-slate-500 tabular-nums">
                      {c.skills.length}{c.skills.length !== c.total ? ` из ${c.total}` : ""}
                    </span>
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-3 pt-1 flex flex-wrap gap-1.5">
                      {c.skills.map((s) => (
                        <span
                          key={s}
                          className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-slate-800 text-gray-600 dark:text-slate-300"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
