import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { FileText, RefreshCw, AlertCircle, Search } from "lucide-react";
import { apiFetch } from "../../lib/auth";

export function LogsTab() {
  const [lines, setLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const q = level === "all" ? "?lines=400" : `?lines=400&level=${level}`;
      const r = await apiFetch(`/api/admin/logs/file${q}`);
      if (!r.ok) {
        setError(`Ошибка загрузки логов: ${r.status}`);
        return;
      }
      const data = await r.json();
      setLines(data.lines ?? []);
    } catch (e: any) {
      setError(e.message ?? "Ошибка сети");
    } finally {
      setLoading(false);
    }
  }, [level]);

  useEffect(() => {
    loadLogs();
    if (!autoRefresh) return;
    const interval = setInterval(loadLogs, 10000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadLogs]);

  const filtered = search.trim()
    ? lines.filter((l) => l.toLowerCase().includes(search.toLowerCase()))
    : lines;

  const levelColor = (line: string) => {
    if (line.includes("level=error") || line.startsWith("ERROR ")) return "text-red-600";
    if (line.includes("level=warning") || line.startsWith("WARNING ") || line.startsWith("WARN ")) return "text-amber-600";
    if (line.includes("level=debug")) return "text-gray-400";
    return "text-gray-700";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Логи</h2>
          <p className="text-sm text-gray-500">Системный лог бэкенда (logs/backend.log)</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="rounded" />
            Автообновление
          </label>
          <Button variant="outline" size="sm" onClick={loadLogs} disabled={loading}>
            <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Обновить
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <FileText className="size-5" />
              Хвост backend.log
              <Badge variant="secondary" className="text-xs">{lines.length} строк</Badge>
            </CardTitle>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="size-4 absolute left-2.5 top-2.5 text-gray-400" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Поиск по логам..."
                  className="pl-9 w-56 h-9"
                />
              </div>
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                className="h-9 px-3 rounded-lg border border-gray-300 bg-white text-sm"
              >
                <option value="all">Все уровни</option>
                <option value="error">Ошибки</option>
                <option value="warning">Предупреждения</option>
                <option value="info">Info</option>
                <option value="debug">Debug</option>
              </select>
            </div>
          </div>
          <CardDescription>Последние записи системного журнала бэкенда</CardDescription>
        </CardHeader>
        <CardContent>
          {loading && <p className="text-sm text-gray-500">Загрузка...</p>}
          {!loading && filtered.length === 0 && (
            <p className="text-sm text-gray-400 py-8 text-center">Записей не найдено</p>
          )}
          <pre className="p-4 bg-gray-950 rounded-lg overflow-x-auto text-xs leading-5 max-h-[70vh] overflow-y-auto font-mono whitespace-pre-wrap break-words">
            {filtered.map((line, i) => (
              <div key={i} className={levelColor(line)}>{line}</div>
            ))}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
