import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import {
  AlertCircle, RefreshCw, Users, FileText, Database,
  Upload, Brain, BookOpen,
} from "lucide-react";
import { apiFetch, logAction } from "../../lib/auth";

export function AdminDashboard() {
  const [users, setUsers] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("users");
  const [logFilter, setLogFilter] = useState("all");
  const [seedLoading, setSeedLoading] = useState(false);
  const [embLoading, setEmbLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [backupLoading, setBackupLoading] = useState(false);
  const [backupMsg, setBackupMsg] = useState("");
  const [extLoading, setExtLoading] = useState(false);
  const [seedMsg, setSeedMsg] = useState("");
  const [embMsg, setEmbMsg] = useState("");
  const [exportMsg, setExportMsg] = useState("");
  const [extMsg, setExtMsg] = useState("");
  const [impMsg, setImpMsg] = useState("");
  const [newUserEmail, setNewUserEmail] = useState("");
  const [newUserPass, setNewUserPass] = useState("");
  const [newUserRole, setNewUserRole] = useState("teacher");
  const [newUserName, setNewUserName] = useState("");
  const [userCreated, setUserCreated] = useState("");
  const [importJson, setImportJson] = useState("");

  const loadData = async () => {
    setLoading(true); setError(null);
    try {
      const [uRes, lRes] = await Promise.all([
        apiFetch("/api/admin/users"),
        apiFetch("/api/admin/logs?limit=200"),
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

  const callAction = async (url: string, body: any, setMsg: (s: string) => void, setLoad: (b: boolean) => void) => {
    setLoad(true); setMsg(""); logAction(url);
    try {
      const r = await apiFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      setMsg(d.message || d.status || "Done");
    } catch (e: any) {
      setMsg("Error: " + e.message);
    } finally {
      setLoad(false);
    }
  };

  const callExport = async () => {
    logAction("/api/admin/export/db");
    setExportLoading(true); setExportMsg("");
    try {
      const r = await apiFetch("/api/admin/export/db", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const d = await r.json();
      setExportMsg(d.message || "Done");
    } catch (e: any) { setExportMsg("Error: " + e.message); }
    finally { setExportLoading(false); }
  };

  const createUser = async () => {
    setUserCreated("");
    try {
      const r = await apiFetch("/api/admin/users/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: newUserEmail, password: newUserPass, role: newUserRole, name: newUserName }),
      });
      const d = await r.json();
      setUserCreated(d.status === "ok" ? `User created: ${d.email}` : "Failed");
      if (d.status === "ok") { setNewUserEmail(""); setNewUserPass(""); setNewUserName(""); }
    } catch (e: any) {
      setUserCreated("Error: " + e.message);
    }
  };

  const importStudents = async () => {
    setImpMsg("");
    try {
      const data = JSON.parse(importJson);
      const r = await apiFetch("/api/admin/students/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const d = await r.json();
      setImpMsg(`Imported: ${d.imported || "?"} students`);
    } catch (e: any) {
      setImpMsg("Error: " + e.message);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Администрирование</h2>
          <p className="text-sm text-gray-500">Управление системой</p>
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
        <TabsList className="flex-wrap">
          <TabsTrigger value="users"><Users className="size-4 mr-2" />Пользователи</TabsTrigger>
          <TabsTrigger value="logs"><FileText className="size-4 mr-2" />Логи</TabsTrigger>
          <TabsTrigger value="db"><Database className="size-4 mr-2" />БД</TabsTrigger>
          <TabsTrigger value="import"><Upload className="size-4 mr-2" />Импорт</TabsTrigger>
          <TabsTrigger value="skills"><Brain className="size-4 mr-2" />Навыки</TabsTrigger>
        </TabsList>

        {/* ── Users tab ── */}
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

          <Card>
            <CardHeader><CardTitle className="text-lg">Создать пользователя</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Input placeholder="Email" value={newUserEmail} onChange={(e) => setNewUserEmail(e.target.value)} />
              <Input placeholder="Password" type="password" value={newUserPass} onChange={(e) => setNewUserPass(e.target.value)} />
              <Input placeholder="Full name" value={newUserName} onChange={(e) => setNewUserName(e.target.value)} />
              <select value={newUserRole} onChange={(e) => setNewUserRole(e.target.value)}
                className="w-full h-9 px-3 rounded-lg border border-gray-300 bg-white text-sm">
                <option value="teacher">Преподаватель</option>
                <option value="admin">Администратор</option>
              </select>
              <Button onClick={createUser}>Создать</Button>
              {userCreated && <p className="text-sm text-green-600">{userCreated}</p>}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Logs tab ── */}
        <TabsContent value="logs" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">История запросов</CardTitle>
                <select value={logFilter} onChange={(e) => setLogFilter(e.target.value)}
                  className="h-9 px-3 rounded-lg border border-gray-300 bg-white text-sm">
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

        {/* ── DB tab ── */}
        <TabsContent value="db" className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-lg"><Database className="size-4 inline mr-2" />Управление БД</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/db/seed", { drop: false }, setSeedMsg, setSeedLoading)} disabled={seedLoading}>
                  {seedLoading ? "..." : "Seed DB"}
                </Button>
                <Button variant="outline" onClick={() => callAction("/api/admin/db/seed", { drop: true }, setSeedMsg, setSeedLoading)} disabled={seedLoading}>
                  Drop + Seed
                </Button>
                {seedMsg && <span className="text-sm text-gray-600">{seedMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/embeddings/generate", { force: false }, setEmbMsg, setEmbLoading)} disabled={embLoading}>
                  <Brain className="size-4 mr-2" />{embLoading ? "..." : "Generate embeddings"}
                </Button>
                <Button variant="outline" onClick={() => callAction("/api/admin/embeddings/generate", { force: true }, setEmbMsg, setEmbLoading)} disabled={embLoading}>
                  Force regenerate
                </Button>
                {embMsg && <span className="text-sm text-gray-600">{embMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={callExport} disabled={exportLoading}>
                  <FileText className="size-4 mr-2" />{exportLoading ? "..." : "Export DB → JSON"}
                </Button>
                {exportMsg && <span className="text-sm text-gray-600">{exportMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/db/backup", {}, setBackupMsg, setBackupLoading)} disabled={backupLoading}>
                  {backupLoading ? "..." : "Backup DB (pg_dump)"}
                </Button>
                {backupMsg && <span className="text-sm text-gray-600">{backupMsg}</span>}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Import tab ── */}
        <TabsContent value="import" className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-lg"><Upload className="size-4 inline mr-2" />Импорт студентов</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-gray-500">JSON-массив студентов:</p>
              <textarea
                value={importJson}
                onChange={(e) => setImportJson(e.target.value)}
                rows={8}
                className="w-full p-3 rounded-lg border border-gray-300 text-sm font-mono"
                placeholder={`[
  {"full_name":"Иванов Иван","group_name":"ИСИТ-31","skills":"python,sql"}
]`}
              />
              <Button onClick={importStudents} disabled={!importJson}>Импортировать</Button>
              {impMsg && <p className="text-sm text-green-600">{impMsg}</p>}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Skills tab ── */}
        <TabsContent value="skills" className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-lg"><Brain className="size-4 inline mr-2" />Расширение таксономии</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-gray-500">Анализ вакансий и добавление новых навыков в it_skills.json</p>
              <Button onClick={() => callAction("/api/admin/skills/extend", { yes: true }, setExtMsg, setExtLoading)} disabled={extLoading}>
                {extLoading ? "..." : "Анализировать и добавить"}
              </Button>
              {extMsg && <p className="text-sm text-green-600">{extMsg}</p>}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
