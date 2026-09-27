import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Shield, Users, FileText, Activity, BookOpen, Loader2 } from "lucide-react";
import { apiFetch } from "../../lib/auth";
import { SelfProfileEditor } from "./SelfProfileEditor";

type AccountProps = { displayName?: string; email?: string };

function AccountCard({ displayName, email, roleLabel }: AccountProps & { roleLabel: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Учётная запись</CardTitle>
        <CardDescription>{roleLabel}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-sm text-gray-900 dark:text-slate-100">
        <p>
          <span className="text-gray-500 dark:text-slate-400">ФИО: </span>
          {displayName || "—"}
        </p>
        <p>
          <span className="text-gray-500 dark:text-slate-400">Email: </span>
          {email || "—"}
        </p>
      </CardContent>
    </Card>
  );
}

/** Полная страница профиля администратора: учётка + ссылки на админ-разделы. */
export function AdminProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  return (
    <div className="space-y-6">
      <AccountCard displayName={displayName} email={email} roleLabel="Администратор" />
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="size-5 text-blue-600" />
            Администрирование
          </CardTitle>
          <CardDescription>Быстрые ссылки на административные разделы</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => onNavigate("admin")}>
            <Users className="mr-2 size-4" />
            Пользователи
          </Button>
          <Button variant="outline" onClick={() => onNavigate("logs")}>
            <FileText className="mr-2 size-4" />
            Логи
          </Button>
          <Button variant="outline" onClick={() => onNavigate("monitoring")}>
            <Activity className="mr-2 size-4" />
            Мониторинг
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

type DisciplineRow = {
  name: string;
  competencies_count?: number;
  skills_count?: number;
  semester?: number | null;
  in_scope?: boolean;
};

/** Полная страница профиля преподавателя: учётка + зона преподавания (read-only). */
export function TeacherProfilePage({ displayName, email }: AccountProps) {
  const [discs, setDiscs] = useState<DisciplineRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch("/api/teacher/krm/disciplines?dir_code=09.03.02")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => setDiscs(Array.isArray(d) ? d : []))
      .catch((e: any) => setError(e?.message || "Не удалось загрузить дисциплины"));
  }, []);

  return (
    <div className="space-y-6">
      <AccountCard displayName={displayName} email={email} roleLabel="Преподаватель" />
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BookOpen className="size-5 text-emerald-600" />
            Зона преподавания
          </CardTitle>
          <CardDescription>Дисциплины направления 09.03.02 (только просмотр)</CardDescription>
        </CardHeader>
        <CardContent>
          {discs === null && !error && (
            <p className="flex items-center gap-2 text-sm text-gray-500 dark:text-slate-400">
              <Loader2 className="size-4 animate-spin" />
              Загрузка…
            </p>
          )}
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
          {discs !== null && !error && (
            discs.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-slate-400">Дисциплины не найдены.</p>
            ) : (
              <ul className="divide-y divide-gray-100 dark:divide-slate-800">
                {discs.map((d) => (
                  <li key={d.name} className="py-2 flex items-center justify-between gap-3">
                    <span className="text-sm text-gray-900 dark:text-slate-100">{d.name}</span>
                    <span className="flex items-center gap-1.5 shrink-0">
                      {typeof d.semester === "number" && (
                        <Badge variant="outline" className="text-xs">сем. {d.semester}</Badge>
                      )}
                      {typeof d.competencies_count === "number" && (
                        <Badge variant="secondary" className="text-xs">компетенций: {d.competencies_count}</Badge>
                      )}
                      {typeof d.skills_count === "number" && (
                        <Badge variant="secondary" className="text-xs">навыков: {d.skills_count}</Badge>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            )
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/** Полная страница профиля студента: редактор своего профиля + ссылка на компетенции. */
export function StudentProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  return (
    <div className="space-y-6">
      <SelfProfileEditor displayName={displayName} email={email} />
      <div>
        <Button variant="outline" onClick={() => onNavigate("data")}>
          <BookOpen className="mr-2 size-4" />
          Мои компетенции
        </Button>
      </div>
    </div>
  );
}
