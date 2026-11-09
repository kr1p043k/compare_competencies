import { useEffect, useState } from "react";
import { api } from "../api";
import { GraduationCap } from "lucide-react";
import { PageHeader } from "./PageHeader";
import { Badge, Button, Card, tableHeadClass, tableRowClass } from "../../design";
import { cn } from "./ui/utils";

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
  // Бэкенд отдаёт признак «профиль не заполнялся»; в типе его не было, из-за
  // чего tsc падал на проверке detail.has_profile === false.
  has_profile?: boolean;
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
      const d = await api<{ students: StudentRow[] }>("/teacher/students");
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
      const d = await api<StudentDetail>(`/admin/students/skills?email=${encodeURIComponent(email)}`);
      setDetail(d as StudentDetail);
      setDetailState("ready");
    } catch (e: any) {
      setDetailErr(errText(e));
      setDetailState("error");
    }
  }

  const owned = new Set((detail?.user_added || []).map((s) => s.toLowerCase()));

  return (
    <div className="space-y-6">
      <PageHeader
        icon={GraduationCap}
        title="Студенты"
        description="Навыки студентов из загруженных профилей и сравнение с требованиями рынка."
        actions={
          <Button variant="secondary" size="sm" onClick={loadList}>
            Обновить
          </Button>
        }
      />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="p-4">
          {state === "loading" && (
            <p className="text-sm text-muted-foreground">Загрузка списка студентов…</p>
          )}
          {state === "error" && (
            <div role="alert" className="text-sm text-destructive">
              Ошибка: {err}
              <button
                onClick={loadList}
                className="ml-2 cursor-pointer rounded-sm underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                Повторить
              </button>
            </div>
          )}
          {state === "empty" && (
            <p className="text-sm text-muted-foreground">
              Студентов с ролью student пока нет. Учётная запись появится здесь после создания пользователя с
              ролью student.
            </p>
          )}
          {state === "ready" && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className={cn("border-b text-left", tableHeadClass)}>
                    <th className="pb-2 pr-3">ФИО / email</th>
                    <th className="pb-2 pr-3 text-right">Навыков</th>
                    <th className="pb-2 pr-3 text-right">Компетенций</th>
                    <th className="pb-2">Уровень</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.email}
                      onClick={() => selectStudent(r.email)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          selectStudent(r.email);
                        }
                      }}
                      tabIndex={0}
                      aria-label={`Показать профиль: ${r.full_name || r.email}`}
                      className={cn(
                        tableRowClass,
                        "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                        selected === r.email && "bg-primary-soft",
                      )}
                    >
                      <td className="py-2 pr-3">
                        <div className="font-medium text-foreground">{r.full_name || "-"}</div>
                        <div className="text-xs text-muted-foreground">{r.email}</div>
                      </td>
                      <td className="py-2 pr-3 text-right text-foreground">
                        {r.has_profile ? r.skills_count : "-"}
                      </td>
                      <td className="py-2 pr-3 text-right text-foreground">
                        {r.has_profile ? r.competencies_count : "-"}
                      </td>
                      <td className="py-2 text-xs text-muted-foreground">{r.target_level || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card className="p-4">
          <h2 className="text-lg font-semibold mb-3 text-foreground">Профиль студента</h2>
          {detailState === "idle" && (
            <p className="text-sm text-muted-foreground">
              Выберите студента из таблицы, чтобы увидеть его навыки и компетенции.
            </p>
          )}
          {detailState === "loading" && (
            <p className="text-sm text-muted-foreground">Загрузка профиля…</p>
          )}
          {detailState === "error" && (
            <p role="alert" className="text-sm text-destructive">
              Ошибка: {detailErr}
            </p>
          )}
          {detailState === "ready" && detail && (
            <div className="space-y-4">
              <div>
                <div className="font-medium text-foreground">{detail.full_name || detail.email}</div>
                <div className="text-xs text-muted-foreground">
                  {detail.email} · уровень: {detail.target_level || "-"}
                  {detail.updated_at
                    ? ` · обновлён: ${new Date(detail.updated_at).toLocaleString("ru-RU")}`
                    : ""}
                </div>
              </div>
              {detail.has_profile === false && (
                <p className="rounded-md bg-warning-soft px-3 py-2 text-sm text-warning">
                  Студент ещё не заполнял профиль: навыков и компетенций нет.
                </p>
              )}
              <div>
                <div className="mb-2 text-sm font-semibold text-foreground">
                  Навыки ({detail.skills.length})
                </div>
                {detail.skills.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Навыков пока нет.</p>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {detail.skills.map((s) => {
                      const isOwn = owned.has(s.toLowerCase());
                      return (
                        <Badge
                          key={s}
                          tone={isOwn ? "success" : "neutral"}
                          title={isOwn ? "добавлен студентом" : "из профиля"}
                          className="rounded text-xs"
                        >
                          {s}
                          {isOwn && " · своё"}
                        </Badge>
                      );
                    })}
                  </div>
                )}
              </div>
              <div>
                <div className="mb-2 text-sm font-semibold text-foreground">
                  Компетенции ({detail.competencies.length})
                </div>
                {detail.competencies.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Компетенций пока нет.</p>
                ) : (
                  <ul className="list-disc space-y-1 pl-5 text-sm text-foreground">
                    {detail.competencies.map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
