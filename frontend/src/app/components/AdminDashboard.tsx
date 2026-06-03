import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import { AlertCircle, RefreshCw, Users, FileText } from "lucide-react";
import { apiFetch } from "../../lib/auth";

export function AdminDashboard() {
  const [users, setUsers] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("users");
  const [logFilter, setLogFilter] = useState("all");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [uRes, lRes] = await Promise.all([
        apiFetch("/api/admin/users"),
        apiFetch(`/api/admin/logs?limit=200`),
      ]);
      if (!uRes.ok || !lRes.ok) throw new Error("Failed to load admin data");
      const uData = await uRes.json();
      const lData = await lRes.json();
      setUsers(uData.users || []);
      setLogs(lData.logs || []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, []);

  const filteredLogs = logFilter === "all" ? logs : logs.filter((l) => l.user === logFilter);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Администрирование</h2>
          <p className="text-sm text-gray-500">Управление пользователями и мониторинг</p>
        </div>
        <Button variant="outline" size="sm" onClick={loadData} disabled={loading}>
          <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Обновить
        </Button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="users"><Users className="size-4 mr-2" />Пользователи</TabsTrigger>
          <TabsTrigger value="logs"><FileText className="size-4 mr-2" />Логи запросов</TabsTrigger>
        </TabsList>

        <TabsContent value="users" className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">Зарегистрированные пользователи</CardTitle></CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-gray-500">
                      <th className="pb-2 font-medium">Имя</th>
                      <th className="pb-2 font-medium">Логин</th>
                      <th className="pb-2 font-medium">Роль</th>
                      <th className="pb-2 font-medium text-right">Запросов</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((u) => (
                      <tr key={u.username} className="border-b border-gray-100">
                        <td className="py-2 font-medium text-gray-900">{u.name}</td>
                        <td className="py-2 text-gray-600">{u.username}</td>
                        <td className="py-2">
                          <Badge variant="outline" className={
                            u.role === "admin" ? "bg-red-50 text-red-700 border-red-200" :
                            u.role === "teacher" ? "bg-blue-50 text-blue-700 border-blue-200" :
                            "bg-green-50 text-green-700 border-green-200"
                          }>
                            {u.role === "admin" ? "Админ" : u.role === "teacher" ? "Преподаватель" : "Студент"}
                          </Badge>
                        </td>
                        <td className="py-2 text-right text-gray-600">{u.total_requests}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="logs" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">История запросов</CardTitle>
                <select value={logFilter} onChange={(e) => setLogFilter(e.target.value)} className="h-9 px-3 rounded-lg border border-gray-300 bg-white text-sm">
                  <option value="all">Все пользователи</option>
                  {users.map((u) => <option key={u.username} value={u.username}>{u.name}</option>)}
                </select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-gray-500 sticky top-0 bg-white">
                      <th className="pb-2 font-medium">Время</th>
                      <th className="pb-2 font-medium">Пользователь</th>
                      <th className="pb-2 font-medium">Метод</th>
                      <th className="pb-2 font-medium">Путь</th>
                      <th className="pb-2 font-medium text-right">Статус</th>
                      <th className="pb-2 font-medium text-right">мс</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLogs.toReversed().slice(0, 200).map((l, i) => (
                      <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-1.5 text-xs text-gray-500 whitespace-nowrap">{new Date(l.timestamp).toLocaleTimeString()}</td>
                        <td className="py-1.5 text-gray-700">{l.user}</td>
                        <td className="py-1.5">
                          <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                            l.method === "GET" ? "bg-green-100 text-green-700" :
                            l.method === "POST" ? "bg-blue-100 text-blue-700" :
                            "bg-gray-100 text-gray-700"
                          }`}>{l.method}</span>
                        </td>
                        <td className="py-1.5 text-xs text-gray-600 max-w-xs truncate">{l.path}</td>
                        <td className="py-1.5 text-right">
                          <span className={`text-xs font-mono ${
                            l.status < 300 ? "text-green-600" : l.status < 400 ? "text-yellow-600" : "text-red-600"
                          }`}>{l.status}</span>
                        </td>
                        <td className="py-1.5 text-right text-xs text-gray-500">{l.duration_ms.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
