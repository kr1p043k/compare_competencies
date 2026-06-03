import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { History, RefreshCw, AlertCircle, MapPin, Briefcase, FileText } from "lucide-react";
import { Button } from "./ui/button";
import { apiFetch } from "../../lib/auth";

export function StudentDashboard() {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await apiFetch("/api/student/history?limit=50");
      if (!res.ok) throw new Error("Failed to load history");
      const data = await res.json();
      setHistory(data.history || []);
    } catch { /* ignore */ } finally { setLoading(false); }
  };

  useEffect(() => {
    loadHistory();
    const handler = () => loadHistory();
    window.addEventListener("student-history-update", handler);
    return () => window.removeEventListener("student-history-update", handler);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Мои запросы</h2>
          <p className="text-sm text-gray-500">История запусков анализа</p>
        </div>
        <Button variant="outline" size="sm" onClick={loadHistory} disabled={loading}>
          <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Обновить
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">История запросов</CardTitle>
        </CardHeader>
        <CardContent>
          {history.length === 0 && !loading && (
            <div className="flex items-center gap-2 text-gray-500 py-4">
              <AlertCircle className="size-5" />
              <span>История пуста. Запустите анализ компетенций, чтобы здесь появились записи.</span>
            </div>
          )}
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-gray-500 sticky top-0 bg-white">
                  <th className="pb-2 font-medium">Время</th>
                  <th className="pb-2 font-medium">Запрос</th>
                  <th className="pb-2 font-medium">Регион</th>
                  <th className="pb-2 font-medium text-right">Вакансий</th>
                  <th className="pb-2 font-medium">Результат</th>
                </tr>
              </thead>
              <tbody>
                {history.toReversed().map((h, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2 text-xs text-gray-500 whitespace-nowrap">{new Date(h.timestamp).toLocaleString()}</td>
                    <td className="py-2">
                      <div className="flex items-center gap-1.5">
                        <Briefcase className="size-3.5 text-gray-400" />
                        <span className="text-gray-900 font-medium">{h.profession || "—"}</span>
                      </div>
                      {h.profile && <span className="text-xs text-gray-400 ml-5">профиль: {h.profile}</span>}
                    </td>
                    <td className="py-2">
                      <div className="flex items-center gap-1.5">
                        <MapPin className="size-3.5 text-gray-400" />
                        <span className="text-gray-600">{h.region === "0" ? "Весь рынок" : h.region || "—"}</span>
                      </div>
                    </td>
                    <td className="py-2 text-right">
                      <Badge variant="secondary" className="text-xs">{h.vacancies_found > 0 ? h.vacancies_found : "—"}</Badge>
                    </td>
                    <td className="py-2">
                      {h.profile ? (
                        <button
                          onClick={() => window.dispatchEvent(new CustomEvent("navigate-analysis", { detail: { profile: h.profile } }))}
                          className="text-blue-600 hover:text-blue-800 underline text-xs cursor-pointer"
                        >
                          <FileText className="size-3.5 inline mr-1" />
                          Открыть анализ
                        </button>
                      ) : (
                        <span className="text-gray-400 text-xs">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
