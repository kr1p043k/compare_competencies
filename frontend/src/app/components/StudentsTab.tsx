import { useEffect, useState } from "react";
import { api } from "../api";
import { useTheme } from "../../lib/theme";

type StudentRow = {
  email: string;
  full_name: string;
  target_level: string | null;
  skills_count: number;
  competencies_count: number;
  has_profile: boolean;
};

type StudentDetail = {
  email: string;
  full_name: string;
  target_level: string;
  skills: string[];
  user_added: string[];
  competencies: string[];
  updated_at: string | null;
};

function errText(e: any): string {
  const raw = e?.message || "неизвестная ошибка";
  try {
    const j = JSON.parse(raw);
    const d = String(j.detail || raw);
    if (/403|forbidden/i.test(raw + d)) return "Нет доступа (нужна роль teacher, rop или admin)";
    return d;
  } catch {
    if (/403|forbidden/i.test(raw)) return "Нет доступа (нужна роль teacher, rop или admin)";
    if (/404/i.test(raw)) return "Профиль не найден";
    return raw;
  }
}

export function StudentsTab() {
  const { theme } = useTheme();
  const dk = theme === "dark";
  const [rows, setRows] = useState<StudentRow[]>([]);
  const [state, setState] = useState<"loading" | "error" | "empty" | "ready">("loading");
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<StudentDetail | null>(null);
  const [detailState, setDetailState] = useState<"idle" | "loading" | "error" | "ready">("idle");
  const [detailErr, setDetailErr] = useState("");

  async function loadList() {
    setState("loading");
    setErr("");
    try {
      const d = await api("/teacher/students");
      const list = (d?.students || []) as StudentRow[];
      setRows(list);
      setState(list.length === 0 ? "empty" : "ready");
    } catch (e: any) {
      setErr(errText(e));
      setState("error");
    }
  }

  useEffect(() => {
    loadList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function selectStudent(email: string) {
    setSelected(email);
    setDetail(null);
    setDetailState("loading");
    setDetailErr("");
    try {
      const d = await api(`/admin/students/skills?email=${encodeURIComponent(email)}`);
      setDetail(d as StudentDetail);
      setDetailState("ready");
    } catch (e: any) {
      setDetailErr(errText(e));
      setDetailState("error");
    }
  }

  const card = `rounded-lg border p-4 ${
    dk ? "border-slate-700 bg-slate-900" : "border-gray-200 bg-white"
  }`;
  const muted = dk ? "text-slate-400" : "text-gray-500";
  const heading = dk ? "text-slate-100" : "text-gray-900";

  const owned = new Set((detail?.user_added || []).map((s) => s.toLowerCase()));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className={card}>
        <div className="flex items-center justify-between mb-3">
          <h2 className={`text-lg font-semibold ${heading}`}>Студенты</h2>
          <button
            onClick={loadList}
            className={`text-xs px-3 py-1.5 rounded-md border cursor-pointer ${
              dk
                ? "border-slate-600 text-slate-200 hover:bg-slate-800"
                : "border-gray-300 text-gray-700 hover:bg-gray-50"
            }`}
          >
            Обновить
          </button>
        </div>
        {state === "loading" && <p className={`text-sm ${muted}`}>Загрузка списка студентов…</p>}
        {state === "error" && (
          <div className="text-sm text-red-600 dark:text-red-300">
            Ошибка: {err}
            <button onClick={loadList} className="ml-2 underline cursor-pointer">
              Повторить
            </button>
          </div>
        )}
        {state === "empty" && (
          <p className={`text-sm ${muted}`}>
            Студентов с ролью student пока нет. Учётная запись появится здесь после создания пользователя с
            ролью student.
          </p>
        )}
        {state === "ready" && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className={`border-b text-left ${dk ? "text-slate-400" : "text-gray-500"}`}>
                  <th className="pb-2 font-medium">ФИО / email</th>
                  <th className="pb-2 font-medium text-right">Навыков</th>
                  <th className="pb-2 font-medium text-right">Компетенций</th>
                  <th className="pb-2 font-medium">Уровень</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.email}
                    onClick={() => selectStudent(r.email)}
                    className={`border-b cursor-pointer ${
                      dk
                        ? "border-slate-800 hover:bg-slate-800"
                        : "border-gray-100 hover:bg-gray-50"
                    } ${selected === r.email ? (dk ? "bg-slate-800" : "bg-blue-50") : ""}`}
                  >
                    <td className="py-2">
                      <div className={`font-medium ${heading}`}>{r.full_name || "—"}</div>
                      <div className={`text-xs ${muted}`}>{r.email}</div>
                    </td>
                    <td className={`py-2 text-right ${heading}`}>{r.has_profile ? r.skills_count : "—"}</td>
                    <td className={`py-2 text-right ${heading}`}>
                      {r.has_profile ? r.competencies_count : "—"}
                    </td>
                    <td className={`py-2 text-xs ${muted}`}>{r.target_level || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className={card}>
        <h2 className={`text-lg font-semibold mb-3 ${heading}`}>Профиль студента</h2>
        {detailState === "idle" && (
          <p className={`text-sm ${muted}`}>Выберите студента из таблицы, чтобы увидеть его навыки и компетенции.</p>
        )}
        {detailState === "loading" && <p className={`text-sm ${muted}`}>Загрузка профиля…</p>}
        {detailState === "error" && <p className="text-sm text-red-600 dark:text-red-300">Ошибка: {detailErr}</p>}
        {detailState === "ready" && detail && (
          <div className="space-y-4">
            <div>
              <div className={`font-medium ${heading}`}>{detail.full_name || detail.email}</div>
              <div className={`text-xs ${muted}`}>
                {detail.email} · уровень: {detail.target_level || "—"}
                {detail.updated_at ? ` · обновлён: ${new Date(detail.updated_at).toLocaleString("ru-RU")}` : ""}
              </div>
            </div>
            <div>
              <div className={`text-sm font-semibold mb-2 ${heading}`}>
                Навыки ({detail.skills.length})
              </div>
              {detail.skills.length === 0 ? (
                <p className={`text-sm ${muted}`}>Навыков пока нет.</p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {detail.skills.map((s) => (
                    <span
                      key={s}
                      title={owned.has(s.toLowerCase()) ? "добавлен студентом" : "из профиля"}
                      className={`inline-block px-2 py-0.5 rounded text-xs ${
                        owned.has(s.toLowerCase())
                          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-800"
                          : "bg-gray-100 text-gray-700 dark:bg-slate-800 dark:text-slate-200"
                      }`}
                    >
                      {s}
                      {owned.has(s.toLowerCase()) && " · своё"}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div>
              <div className={`text-sm font-semibold mb-2 ${heading}`}>
                Компетенции ({detail.competencies.length})
              </div>
              {detail.competencies.length === 0 ? (
                <p className={`text-sm ${muted}`}>Компетенций пока нет.</p>
              ) : (
                <ul className={`text-sm list-disc pl-5 space-y-1 ${dk ? "text-slate-200" : "text-gray-700"}`}>
                  {detail.competencies.map((c) => (
                    <li key={c}>{c}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
