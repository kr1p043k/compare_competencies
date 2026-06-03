import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Badge } from "./ui/badge";
import { BarChart3, RefreshCw, FileText, AlertCircle } from "lucide-react";
import { Button } from "./ui/button";
import { apiFetch } from "../../lib/auth";

export function TeacherDashboard() {
  const [stats, setStats] = useState<{ total_reports: number; by_profession: { profession: string; reports: number }[] } | null>(null);
  const [loading, setLoading] = useState(true);

  const loadStats = async () => {
    setLoading(true);
    try {
      const res = await apiFetch("/api/teacher/stats");
      if (!res.ok) throw new Error("Failed to load stats");
      setStats(await res.json());
    } catch { /* ignore */ } finally { setLoading(false); }
  };

  useEffect(() => { loadStats(); }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Панель преподавателя</h2>
          <p className="text-sm text-gray-500">Статистика отчётов по компетенциям</p>
        </div>
        <Button variant="outline" size="sm" onClick={loadStats} disabled={loading}>
          <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Обновить
        </Button>
      </div>

      {stats && (
        <>
          <Card>
            <CardHeader className="flex flex-row items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <FileText className="size-5 text-white" />
              </div>
              <div>
                <CardTitle className="text-lg">Всего сгенерировано отчётов</CardTitle>
                <CardDescription>За всё время работы системы</CardDescription>
              </div>
              <div className="ml-auto text-3xl font-bold text-blue-600">{stats.total_reports}</div>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">По профессиям</CardTitle>
              <CardDescription>Распределение отчётов по направлениям</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {stats.by_profession.map((p) => {
                  const max = Math.max(...stats.by_profession.map((x) => x.reports), 1);
                  const pct = (p.reports / max) * 100;
                  return (
                    <div key={p.profession} className="flex items-center gap-3">
                      <span className="w-24 text-sm font-medium text-gray-700 truncate">{p.profession}</span>
                      <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="text-sm font-semibold text-gray-900 w-8 text-right">{p.reports}</span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {!stats && !loading && (
        <div className="flex items-center gap-2 text-gray-500 py-8">
          <AlertCircle className="size-5" />
          <span>Нет данных</span>
        </div>
      )}
    </div>
  );
}
