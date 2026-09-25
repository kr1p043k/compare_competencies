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
export function TaxonomyBrowser() {
  const [cats, setCats] = useState<TaxCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});

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
