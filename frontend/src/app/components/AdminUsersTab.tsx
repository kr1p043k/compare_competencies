/**
 * Вкладка «Пользователи» админ-консоли.
 *
 * Серверная пагинация и фильтры (GET /api/admin/users?q=&role=&is_active=&limit=&offset=),
 * карточка пользователя с сессиями (GET /api/admin/users/{id}), действия PATCH/POST,
 * создание пользователя и редактор направлений РОП. Все необратимые действия требуют
 * ввода подтверждения (адрес пользователя или слово УДАЛИТЬ) в диалоге.
 * Действия над другим администратором дополнительно требуют пароль текущего
 * администратора (caller_password), иначе сервер отвечает 403.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Field,
  inputClass,
  Skeleton,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design/primitives";
import {
  asRole,
  type Role,
  fmtDateTime,
  ROLES,
  ROLE_LABEL,
  roleLabel,
  roleTone,
} from "./adminRoles";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import {
  ChevronLeft,
  ChevronRight,
  Ellipsis,
  IdCard,
  KeyRound,
  LogOut,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  ShieldOff,
  UserCog,
} from "lucide-react";

const DEACTIVATION_WORD = "УДАЛИТЬ";

const SKELETON_ROWS = 6;
const selectClass =
  "h-9 px-3 rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-950 text-sm";

interface Direction {
  dir_code: string;
  name: string;
}

interface AdminUser {
  id: string;
  username: string;
  role: Role;
  name: string;
  total_requests: number;
  is_active: boolean;
  directions: string[];
  totp_enabled: boolean;
  open_sessions: number;
  last_activity: string | number | null;
}

interface UsersPage {
  users: AdminUser[];
  total: number;
  limit: number;
  offset: number;
}

interface UserSession {
  id: string;
  ip_address: string;
  user_agent: string;
  logged_in_at: string | number | null;
  last_activity: string | number | null;
  is_open: boolean;
}

interface UserDetail {
  user: AdminUser;
  directions: string[];
  sessions: UserSession[];
  totp_enabled: boolean;
}

type ActionKind = "deactivate" | "password" | "revoke" | "twofa" | "role";

interface ActionDialog {
  kind: ActionKind;
  user: AdminUser;
}

function truncate(text: string, max = 24): string {
  if (!text) return "-";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

const ACTION_META: Record<ActionKind, { title: string; confirm: string }> = {
  deactivate: { title: "Деактивация пользователя", confirm: "Деактивировать" },
  password: { title: "Сброс пароля", confirm: "Сохранить пароль" },
  revoke: { title: "Отзыв сессий", confirm: "Отозвать сессии" },
  twofa: { title: "Сброс 2FA", confirm: "Сбросить 2FA" },
  role: { title: "Смена роли", confirm: "Сохранить роль" },
};

/**
 * Действия, которые сервер выполняет только с подтверждением caller_password,
 * если цель - другой администратор. Отзыв сессий этого не требует.
 */
const STEP_UP_KINDS: Record<ActionKind, boolean> = {
  deactivate: true,
  password: true,
  revoke: false,
  twofa: true,
  role: true,
};

export function AdminUsersTab() {
  // текущий пользователь: без него не определить, что цель действия - другой админ
  const auth = useAuth();
  // список пользователей: серверная пагинация и фильтры
  const [rows, setRows] = useState<AdminUser[]>([]);
  const [total, setTotal] = useState(0);
  const [phase, setPhase] = useState<"loading" | "error" | "ready">("loading");
  const [fetchError, setFetchError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const requestIdRef = useRef(0);
  const detailReqRef = useRef(0);

  const [search, setSearch] = useState("");
  const [q, setQ] = useState("");
  const [roleFilter, setRoleFilter] = useState<"all" | Role>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "inactive">("all");
  const [limit, setLimit] = useState(25);
  const [offset, setOffset] = useState(0);

  // справочники: направления и список РОП
  const [directions, setDirections] = useState<Direction[]>([]);
  const [ropList, setRopList] = useState<AdminUser[]>([]);

  // результаты прямых действий (активация и пр.), живут в aria-live области
  const [banner, setBanner] = useState<{ text: string; ok: boolean } | null>(null);

  // форма создания пользователя
  const [newUserEmail, setNewUserEmail] = useState("");
  const [newUserPass, setNewUserPass] = useState("");
  const [newUserRole, setNewUserRole] = useState<Role>("teacher");
  const [newUserName, setNewUserName] = useState("");
  const [newUserDirs, setNewUserDirs] = useState<string[]>([]);
  const [userCreated, setUserCreated] = useState<{ text: string; ok: boolean } | null>(null);
  const [createBusy, setCreateBusy] = useState(false);

  // редактор направлений РОП
  const [editUser, setEditUser] = useState("");
  const [editUserDirs, setEditUserDirs] = useState<string[]>([]);
  const [dirBusy, setDirBusy] = useState(false);
  const [dirMsg, setDirMsg] = useState<{ text: string; ok: boolean } | null>(null);

  // диалог с подтверждением действия над пользователем
  const [dialog, setDialog] = useState<ActionDialog | null>(null);
  const [confirmText, setConfirmText] = useState("");
  const [newPass1, setNewPass1] = useState("");
  const [newPass2, setNewPass2] = useState("");
  // пароль администратора для шага повышенного подтверждения (живёт только пока открыт диалог)
  const [callerPassword, setCallerPassword] = useState("");
  const [roleValue, setRoleValue] = useState<Role>("student");
  const [busy, setBusy] = useState(false);
  const [dialogMsg, setDialogMsg] = useState("");
  const [dialogErr, setDialogErr] = useState("");

  // карточка пользователя
  const [detailId, setDetailId] = useState<string | null>(null);
  const [detail, setDetail] = useState<UserDetail | null>(null);
  const [detailPhase, setDetailPhase] = useState<"idle" | "loading" | "error" | "ready">("idle");
  const [detailError, setDetailError] = useState("");

  const passOk = newPass1.length >= 10 && newPass1 === newPass2;

  const needWord = dialog?.kind === "deactivate";
  const confirmSatisfied =
    dialog === null
      ? false
      : dialog.kind === "role"
        ? true
        : needWord
          ? confirmText.trim().toUpperCase() === DEACTIVATION_WORD
          : confirmText.trim().toLowerCase() === dialog.user.username.toLowerCase();

  /**
   * Требуется ли пароль текущего администратора: цель имеет роль «admin»
   * и это не сам текущий пользователь (сравниваем email из useAuth).
   * Если email текущего пользователя неизвестен, считаем целью «другого»:
   * лучше лишний раз спросить пароль, чем получить безвыходный 403.
   */
  const stepUpRequired =
    dialog !== null &&
    STEP_UP_KINDS[dialog.kind] &&
    dialog.user.role === "admin" &&
    // Сброс 2FA требует пароль и для своего аккаунта: без step-up угнанная
    // сессия снимает второй фактор одной кнопкой.
    (dialog.kind === "twofa" ||
      auth.username === null ||
      dialog.user.username.trim().toLowerCase() !== auth.username.trim().toLowerCase());

  useEffect(() => {
    const timer = setTimeout(() => {
      setQ(search.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  const loadUsers = useCallback(async () => {
    const reqId = ++requestIdRef.current;
    setPhase("loading");
    setFetchError("");
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (roleFilter !== "all") params.set("role", roleFilter);
    if (statusFilter !== "all") params.set("is_active", statusFilter === "active" ? "true" : "false");
    params.set("limit", String(limit));
    params.set("offset", String(offset));
    try {
      const d = await request<UsersPage>(`/api/admin/users?${params.toString()}`);
      if (reqId !== requestIdRef.current) return;
      const list: AdminUser[] = Array.isArray(d?.users) ? d.users : [];
      const totalCount =
        typeof d?.total === "number" && d.total >= 0 ? d.total : list.length;
      setRows(list);
      setTotal(totalCount);
      setPhase("ready");
      // после деактивации или смены роли текущая страница может опустеть
      if (list.length === 0 && totalCount > 0 && offset > 0) {
        setOffset(Math.max(0, Math.floor((totalCount - 1) / limit) * limit));
      }
    } catch (e) {
      if (reqId !== requestIdRef.current) return;
      setFetchError(e instanceof Error ? e.message : "Не удалось загрузить список");
      setPhase("error");
    }
  }, [q, roleFilter, statusFilter, limit, offset, reloadKey]);

  useEffect(() => {
    void loadUsers();
  }, [loadUsers]);

  // справочники грузятся один раз при монтировании
  useEffect(() => {
    request<unknown>("/api/admin/directions")
      .then((d) => setDirections(Array.isArray(d) ? (d as Direction[]) : []))
      .catch(() => setDirections([]));
    request<UsersPage>("/api/admin/users?role=rop&limit=500")
      .then((d) => setRopList(Array.isArray(d?.users) ? d.users : []))
      .catch(() => setRopList([]));
  }, []);

  const bumpReload = () => setReloadKey((k) => k + 1);

  /* --------------------------------------------------------- действия по строке */

  const openAction = (kind: ActionKind, user: AdminUser) => {
    setDialog({ kind, user });
    setConfirmText("");
    setNewPass1("");
    setNewPass2("");
    setCallerPassword("");
    setRoleValue(user.role);
    setDialogMsg("");
    setDialogErr("");
  };

  const closeDialog = () => {
    setDialog(null);
    setConfirmText("");
    setNewPass1("");
    setNewPass2("");
    setCallerPassword("");
    setDialogMsg("");
    setDialogErr("");
  };

  const activateUser = async (user: AdminUser) => {
    setBusy(true);
    setBanner(null);
    try {
      await request(`/api/admin/users/${user.id}`, { method: "PATCH", body: { is_active: true } });
      setBanner({ text: `Пользователь активирован: ${user.username}`, ok: true });
      bumpReload();
    } catch (e) {
      setBanner({
        text: "Ошибка: " + (e instanceof Error ? e.message : "не удалось активировать"),
        ok: false,
      });
    } finally {
      setBusy(false);
    }
  };

  const submitDialog = async () => {
    if (!dialog || busy) return;
    if (dialog.kind !== "role" && !confirmSatisfied) return;
    if (dialog.kind === "role" && roleValue === dialog.user.role) return;
    if (dialog.kind === "password" && !passOk) return;
    if (stepUpRequired && callerPassword === "") return;
    setBusy(true);
    setDialogErr("");
    setDialogMsg("");
    const { kind, user } = dialog;
    // пароль текущего администратора отправляем только для действий над другим админом
    const stepUp = stepUpRequired ? { caller_password: callerPassword } : {};
    try {
      if (kind === "role") {
        await request(`/api/admin/users/${user.id}`, {
          method: "PATCH",
          body: { role: roleValue, ...stepUp },
        });
        setDialogMsg(`Роль обновлена: ${roleLabel(roleValue)}`);
      } else if (kind === "deactivate") {
        await request(`/api/admin/users/${user.id}`, {
          method: "PATCH",
          body: { is_active: false, ...stepUp },
        });
        setDialogMsg("Пользователь деактивирован");
      } else if (kind === "password") {
        const d = await request<{ status?: string; revoked_sessions?: number }>(
          `/api/admin/users/${user.id}/password`,
          { method: "POST", body: { new_password: newPass1, ...stepUp } },
        );
        setDialogMsg(`Пароль обновлён. Отозвано сессий: ${String(d?.revoked_sessions ?? 0)}`);
      } else if (kind === "revoke") {
        const d = await request<{ status?: string; revoked?: number }>(
          `/api/admin/users/${user.id}/sessions/revoke`,
          { method: "POST" },
        );
        setDialogMsg(`Отозвано сессий: ${String(d?.revoked ?? 0)}`);
      } else if (kind === "twofa") {
        await request(`/api/admin/users/${user.id}/2fa/reset`, {
          method: "POST",
          body: stepUpRequired ? { caller_password: callerPassword } : undefined,
        });
        setDialogMsg("2FA сброшена");
      }
      bumpReload();
    } catch (e) {
      setDialogErr(e instanceof Error ? e.message : "Не удалось выполнить действие");
    } finally {
      setBusy(false);
    }
  };

  /* ------------------------------------------------------- карточка пользователя */

  const loadDetail = useCallback(async (id: string) => {
    // защита от устаревших ответов: карточка A не должна перезаписать карточку B
    const reqId = ++detailReqRef.current;
    setDetailPhase("loading");
    setDetailError("");
    try {
      const d = await request<UserDetail>(`/api/admin/users/${id}`);
      if (reqId !== detailReqRef.current) return;
      setDetail(d);
      setDetailPhase("ready");
    } catch (e) {
      if (reqId !== detailReqRef.current) return;
      setDetailError(e instanceof Error ? e.message : "Не удалось загрузить карточку");
      setDetailPhase("error");
    }
  }, []);

  const openDetail = (id: string) => {
    setDetailId(id);
    setDetail(null);
    void loadDetail(id);
  };

  const closeDetail = () => {
    setDetailId(null);
    setDetail(null);
    setDetailPhase("idle");
    setDetailError("");
  };

  /* ------------------------------------------------------------ создание, РОП */

  const createUser = async () => {
    setUserCreated(null);
    if (!/^\S+@\S+\.\S+$/.test(newUserEmail.trim())) {
      setUserCreated({ text: "Ошибка: укажите корректный адрес электронной почты", ok: false });
      return;
    }
    if (newUserPass.length < 10) {
      setUserCreated({ text: "Ошибка: пароль должен быть не короче 10 символов", ok: false });
      return;
    }
    if (!newUserName.trim()) {
      setUserCreated({ text: "Ошибка: укажите ФИО", ok: false });
      return;
    }
    if (newUserRole === "rop" && newUserDirs.length === 0) {
      setUserCreated({ text: "Ошибка: выберите хотя бы одно направление", ok: false });
      return;
    }
    setCreateBusy(true);
    try {
      const d = await request<{ status: string; email?: string }>("/api/admin/users/create", {
        method: "POST",
        body: {
          email: newUserEmail,
          password: newUserPass,
          role: newUserRole,
          name: newUserName,
          directions: newUserRole === "rop" ? newUserDirs : [],
        },
      });
      setUserCreated({
        text:
          d.status === "ok"
            ? `Пользователь создан: ${d.email ?? newUserEmail}`
            : "Не удалось создать пользователя",
        ok: d.status === "ok",
      });
      if (d.status === "ok") {
        setNewUserEmail("");
        setNewUserPass("");
        setNewUserName("");
        setNewUserDirs([]);
        bumpReload();
      }
    } catch (e) {
      setUserCreated({
        text: "Ошибка: " + (e instanceof Error ? e.message : "неизвестная ошибка"),
        ok: false,
      });
    } finally {
      setCreateBusy(false);
    }
  };

  const saveDirections = async () => {
    if (!editUser) return;
    setDirBusy(true);
    setDirMsg(null);
    try {
      const d = await request<{ status: string }>(`/api/admin/users/${editUser}/directions`, {
        method: "PUT",
        body: { directions: editUserDirs },
      });
      setDirMsg({ text: d.status === "ok" ? "Сохранено" : "Не сохранено", ok: d.status === "ok" });
    } catch (e) {
      setDirMsg({
        text: "Ошибка: " + (e instanceof Error ? e.message : "неизвестная ошибка"),
        ok: false,
      });
    } finally {
      setDirBusy(false);
    }
  };

  /* ------------------------------------------------------------------ пагинация */

  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + limit, total);
  const hasFilters = q !== "" || roleFilter !== "all" || statusFilter !== "all";
  const showSkeleton = phase === "loading" && rows.length === 0;

  return (
    <div className="space-y-4">
      {banner && (
        <p
          role="status"
          aria-live="polite"
          className={
            banner.ok
              ? "rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-300"
              : "rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
          }
        >
          {banner.text}
        </p>
      )}

      <Card>
        <CardHeader
          title="Пользователи"
          description="Поиск, роли, активные сессии и двухфакторная аутентификация."
          actions={
            <Button
              size="sm"
              variant="secondary"
              onClick={bumpReload}
              loading={phase === "loading"}
              iconLeft={<RefreshCw className="size-4" />}
            >
              Обновить
            </Button>
          }
        />
        <CardBody className="space-y-4">
          {/* фильтры: всё считается на сервере */}
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Поиск" htmlFor="users-search" className="w-64 max-w-full">
              <input
                id="users-search"
                type="search"
                className={inputClass}
                placeholder="Имя или email"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </Field>
            <Field label="Роль" htmlFor="users-role">
              <select
                id="users-role"
                className={selectClass}
                value={roleFilter}
                onChange={(e) => {
                  setRoleFilter(e.target.value as "all" | Role);
                  setOffset(0);
                }}
              >
                <option value="all">Все роли</option>
                <option value="admin">Админ</option>
                <option value="teacher">Преподаватель</option>
                <option value="rop">РОП</option>
                <option value="student">Студент</option>
              </select>
            </Field>
            <Field label="Статус" htmlFor="users-status">
              <select
                id="users-status"
                className={selectClass}
                value={statusFilter}
                onChange={(e) => {
                  setStatusFilter(e.target.value as "all" | "active" | "inactive");
                  setOffset(0);
                }}
              >
                <option value="all">Все</option>
                <option value="active">Активные</option>
                <option value="inactive">Деактивированные</option>
              </select>
            </Field>
            <Field label="На странице" htmlFor="users-limit">
              <select
                id="users-limit"
                className={selectClass}
                value={String(limit)}
                onChange={(e) => {
                  setLimit(Number(e.target.value) || 25);
                  setOffset(0);
                }}
              >
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </select>
            </Field>
          </div>

          {phase === "error" && rows.length === 0 && (
            <ErrorState
              title="Не удалось загрузить список пользователей"
              description={fetchError}
              onRetry={loadUsers}
            />
          )}
          {phase === "error" && rows.length > 0 && (
            <div
              role="alert"
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
            >
              <span>{fetchError}</span>
              <Button size="sm" variant="secondary" onClick={loadUsers}>
                Повторить
              </Button>
            </div>
          )}

          {phase === "ready" && rows.length === 0 && (
            <EmptyState
              title={hasFilters ? "Пользователи не найдены" : "Пользователей нет"}
              description={
                hasFilters
                  ? "Попробуйте изменить условия поиска или сбросить фильтры."
                  : "Создайте первого пользователя в форме ниже."
              }
              action={
                hasFilters ? (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      setSearch("");
                      setRoleFilter("all");
                      setStatusFilter("all");
                      setOffset(0);
                    }}
                  >
                    Сбросить фильтры
                  </Button>
                ) : undefined
              }
            />
          )}

          {(showSkeleton || rows.length > 0) && (
            <TableFrame>
              <table className="w-full text-sm">
                <thead className={tableHeadClass}>
                  <tr>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Имя</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Email</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Роль</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Направления</th>
                    <th scope="col" className="px-3 py-2.5 text-right font-semibold">Сессии</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">2FA</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Активность</th>
                    <th scope="col" className="px-3 py-2.5 text-right font-semibold">Запросов</th>
                    <th scope="col" className="px-3 py-2.5 text-right font-semibold">
                      <span className="sr-only">Действия</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {showSkeleton
                    ? Array.from({ length: SKELETON_ROWS }).map((_, i) => (
                        <tr key={i} className={tableRowClass}>
                          {Array.from({ length: 9 }).map((__, j) => (
                            <td key={j} className="px-3 py-3">
                              <Skeleton className="h-4 w-16" />
                            </td>
                          ))}
                        </tr>
                      ))
                    : rows.map((u) => (
                        <tr key={u.id} className={tableRowClass}>
                          <td className="px-3 py-2 font-medium text-foreground">
                            <span className={u.is_active ? "" : "opacity-60"}>
                              {u.name || "-"}
                            </span>
                            {!u.is_active && (
                              <Badge tone="danger" className="ml-2">
                                деактивирован
                              </Badge>
                            )}
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">{u.username}</td>
                          <td className="px-3 py-2">
                            <Badge tone={roleTone(u.role)}>{roleLabel(u.role)}</Badge>
                          </td>
                          <td className="px-3 py-2">
                            {(u.directions ?? []).length > 0 ? (
                              <span className="inline-flex flex-wrap items-center gap-1">
                                {u.directions.slice(0, 3).map((d) => (
                                  <span
                                    key={d}
                                    className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground"
                                  >
                                    {d}
                                  </span>
                                ))}
                                {u.directions.length > 3 && (
                                  <span className="text-xs text-muted-foreground">
                                    +{u.directions.length - 3}
                                  </span>
                                )}
                              </span>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-foreground">
                            {u.open_sessions}
                          </td>
                          <td className="px-3 py-2">
                            <Badge tone={u.totp_enabled ? "success" : "neutral"}>
                              {u.totp_enabled ? "вкл" : "выкл"}
                            </Badge>
                          </td>
                          <td className="whitespace-nowrap px-3 py-2 text-xs text-muted-foreground">
                            {fmtDateTime(u.last_activity)}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                            {u.total_requests}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <DropdownMenu>
                              <DropdownMenuTrigger
                                aria-label={`Действия: ${u.name} (${u.username})`}
                                className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                              >
                                <Ellipsis className="size-4" />
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-52">
                                <DropdownMenuItem onSelect={() => openDetail(u.id)}>
                                  <IdCard className="size-4" /> Карточка
                                </DropdownMenuItem>
                                <DropdownMenuItem onSelect={() => openAction("role", u)}>
                                  <UserCog className="size-4" /> Сменить роль
                                </DropdownMenuItem>
                                {u.is_active ? (
                                  <DropdownMenuItem
                                    variant="destructive"
                                    onSelect={() => openAction("deactivate", u)}
                                  >
                                    <ShieldOff className="size-4" /> Деактивировать
                                  </DropdownMenuItem>
                                ) : (
                                  <DropdownMenuItem onSelect={() => void activateUser(u)}>
                                    <ShieldCheck className="size-4" /> Активировать
                                  </DropdownMenuItem>
                                )}
                                <DropdownMenuSeparator />
                                <DropdownMenuItem onSelect={() => openAction("password", u)}>
                                  <KeyRound className="size-4" /> Сбросить пароль
                                </DropdownMenuItem>
                                <DropdownMenuItem onSelect={() => openAction("revoke", u)}>
                                  <LogOut className="size-4" /> Отозвать сессии
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onSelect={() => openAction("twofa", u)}
                                  disabled={!u.totp_enabled}
                                >
                                  <RotateCcw className="size-4" /> Сбросить 2FA
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </TableFrame>
          )}

          {total > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground" aria-live="polite">
                Показано {first}-{last} из {total}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={offset === 0 || phase === "loading"}
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                >
                  <ChevronLeft className="size-4" /> Назад
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={offset + limit >= total || phase === "loading"}
                  onClick={() => setOffset(offset + limit)}
                >
                  Вперёд <ChevronRight className="size-4" />
                </Button>
              </div>
            </div>
          )}
        </CardBody>
      </Card>

      {/* Создание пользователя */}
      <Card>
        <CardHeader
          title="Создать пользователя"
          description="Адрес, стартовый пароль (не короче 10 символов) и роль."
        />
        <CardBody>
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              void createUser();
            }}
          >
            <Field label="Электронная почта" htmlFor="new-user-email">
              <input
                id="new-user-email"
                type="email"
                className={inputClass}
                value={newUserEmail}
                onChange={(e) => setNewUserEmail(e.target.value)}
                autoComplete="off"
              />
            </Field>
            <Field label="Пароль" htmlFor="new-user-pass" hint="Минимум 10 символов">
              <input
                id="new-user-pass"
                type="password"
                className={inputClass}
                value={newUserPass}
                onChange={(e) => setNewUserPass(e.target.value)}
                autoComplete="new-password"
              />
            </Field>
            <Field label="ФИО" htmlFor="new-user-name">
              <input
                id="new-user-name"
                className={inputClass}
                value={newUserName}
                onChange={(e) => setNewUserName(e.target.value)}
                autoComplete="off"
              />
            </Field>
            <Field label="Роль" htmlFor="new-user-role">
              <select
                id="new-user-role"
                className={`${selectClass} w-full`}
                value={newUserRole}
                onChange={(e) => {
                  setNewUserRole(asRole(e.target.value));
                  setNewUserDirs([]);
                }}
              >
                <option value="teacher">Преподаватель</option>
                <option value="admin">Администратор</option>
                <option value="rop">Руководитель ОП</option>
              </select>
            </Field>
            {newUserRole === "rop" && (
              <Field label="Направления" htmlFor="new-user-dirs">
                <div
                  id="new-user-dirs"
                  className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-border p-3"
                >
                  {directions.map((d) => (
                    <label key={d.dir_code} className="flex cursor-pointer items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={newUserDirs.includes(d.dir_code)}
                        onChange={(e) =>
                          setNewUserDirs((prev) =>
                            e.target.checked
                              ? [...prev, d.dir_code]
                              : prev.filter((x) => x !== d.dir_code),
                          )
                        }
                        className="accent-violet-600"
                      />
                      <span className="font-mono text-xs text-muted-foreground">{d.dir_code}</span>
                      <span className="truncate text-xs text-muted-foreground">{d.name}</span>
                    </label>
                  ))}
                  {directions.length === 0 && (
                    <p className="text-xs text-muted-foreground">Направления не загружены</p>
                  )}
                </div>
              </Field>
            )}
            <div className="flex items-center gap-3">
              <Button type="submit" loading={createBusy}>
                Создать
              </Button>
              {userCreated && (
                <p
                  role="status"
                  aria-live="polite"
                  className={
                    userCreated.ok
                      ? "text-sm text-emerald-600 dark:text-emerald-400"
                      : "text-sm text-red-600 dark:text-red-400"
                  }
                >
                  {userCreated.text}
                </p>
              )}
            </div>
          </form>
        </CardBody>
      </Card>

      {/* Редактор направлений РОП */}
      <Card>
        <CardHeader
          title="Направления РОП"
          description="Настройте, какие направления доступны руководителю ОП."
        />
        <CardBody className="space-y-3">
          <Field label="Пользователь" htmlFor="rop-user-select">
            <select
              id="rop-user-select"
              className={`${selectClass} w-full`}
              value={editUser}
              onChange={(e) => {
                const u = ropList.find((x) => x.id === e.target.value);
                setEditUser(e.target.value);
                setEditUserDirs(u?.directions ?? []);
                setDirMsg(null);
              }}
            >
              <option value="">Выберите пользователя</option>
              {ropList.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.name} ({u.username})
                </option>
              ))}
            </select>
          </Field>
          {editUser && (
            <>
              <div className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-border p-3">
                {directions.map((d) => (
                  <label key={d.dir_code} className="flex cursor-pointer items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={editUserDirs.includes(d.dir_code)}
                      onChange={(e) =>
                        setEditUserDirs((prev) =>
                          e.target.checked
                            ? [...prev, d.dir_code]
                            : prev.filter((x) => x !== d.dir_code),
                        )
                      }
                      className="accent-violet-600"
                    />
                    <span className="font-mono text-xs text-muted-foreground">{d.dir_code}</span>
                    <span className="truncate text-xs text-muted-foreground">{d.name}</span>
                  </label>
                ))}
                {directions.length === 0 && (
                  <p className="text-xs text-muted-foreground">Направления не загружены</p>
                )}
              </div>
              <div className="flex items-center gap-3">
                <Button onClick={() => void saveDirections()} loading={dirBusy}>
                  Сохранить
                </Button>
                {dirMsg && (
                  <p
                    role="status"
                    aria-live="polite"
                    className={
                      dirMsg.ok
                        ? "text-sm text-emerald-600 dark:text-emerald-400"
                        : "text-sm text-red-600 dark:text-red-400"
                    }
                  >
                    {dirMsg.text}
                  </p>
                )}
              </div>
            </>
          )}
        </CardBody>
      </Card>

      {/* Диалог подтверждения действия */}
      <Dialog open={dialog !== null} onOpenChange={(open) => !open && closeDialog()}>
        <DialogContent className="sm:max-w-md">
          {dialog && (
            <>
              <DialogHeader>
                <DialogTitle>{ACTION_META[dialog.kind].title}</DialogTitle>
                <DialogDescription>
                  {dialog.kind === "deactivate" &&
                    `${dialog.user.name} (${dialog.user.username}) потеряет доступ к системе. Войти будет нельзя, пока аккаунт не активируют снова.`}
                  {dialog.kind === "password" &&
                    `Задайте новый пароль для ${dialog.user.username}. Все активные сессии будут отозваны.`}
                  {dialog.kind === "revoke" &&
                    `Все активные сессии пользователя ${dialog.user.username} будут завершены.`}
                  {dialog.kind === "twofa" &&
                    `Двухфакторная аутентификация пользователя ${dialog.user.username} будет сброшена.`}
                  {dialog.kind === "role" && `Выберите новую роль для ${dialog.user.username}.`}
                </DialogDescription>
              </DialogHeader>
              <form
                className="space-y-4"
                onSubmit={(e) => {
                  e.preventDefault();
                  void submitDialog();
                }}
              >
                {dialog.kind === "role" && (
                  <Field label="Роль" htmlFor="dialog-role">
                    <select
                      id="dialog-role"
                      className={`${selectClass} w-full`}
                      value={roleValue}
                      onChange={(e) => setRoleValue(asRole(e.target.value))}
                    >
                      {ROLES.map((r) => (
                        <option key={r} value={r}>
                          {ROLE_LABEL[r]}
                        </option>
                      ))}
                    </select>
                  </Field>
                )}
                {dialog.kind === "password" && (
                  <>
                    <Field
                      label="Новый пароль"
                      htmlFor="dialog-pass-1"
                      hint="Минимум 10 символов"
                      error={newPass1.length > 0 && newPass1.length < 10 ? "Минимум 10 символов" : undefined}
                    >
                      <input
                        id="dialog-pass-1"
                        type="password"
                        className={inputClass}
                        value={newPass1}
                        onChange={(e) => setNewPass1(e.target.value)}
                        autoComplete="new-password"
                      />
                    </Field>
                    <Field
                      label="Повторите пароль"
                      htmlFor="dialog-pass-2"
                      error={newPass2.length > 0 && newPass1 !== newPass2 ? "Пароли не совпадают" : undefined}
                    >
                      <input
                        id="dialog-pass-2"
                        type="password"
                        className={inputClass}
                        value={newPass2}
                        onChange={(e) => setNewPass2(e.target.value)}
                        autoComplete="new-password"
                      />
                    </Field>
                  </>
                )}
                {dialog.kind !== "role" && (
                  <Field
                    label={
                      needWord
                        ? "Введите слово УДАЛИТЬ для подтверждения"
                        : `Введите адрес ${dialog.user.username} для подтверждения`
                    }
                    htmlFor="dialog-confirm"
                  >
                    <input
                      id="dialog-confirm"
                      className={inputClass}
                      value={confirmText}
                      onChange={(e) => setConfirmText(e.target.value)}
                      aria-invalid={!confirmSatisfied || undefined}
                      autoComplete="off"
                      autoFocus
                    />
                  </Field>
                )}
                {stepUpRequired && (
                  <Field
                    label="Пароль администратора"
                    htmlFor="dialog-caller-password"
                    hint="Нужен для подтверждения действий с другим администратором"
                  >
                    <input
                      id="dialog-caller-password"
                      type="password"
                      className={inputClass}
                      value={callerPassword}
                      onChange={(e) => setCallerPassword(e.target.value)}
                      autoComplete="current-password"
                      required
                      aria-invalid={callerPassword === "" || undefined}
                    />
                  </Field>
                )}
                {dialogErr && (
                  <div
                    role="alert"
                    className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
                  >
                    {dialogErr}
                  </div>
                )}
                {dialogMsg && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-300"
                  >
                    {dialogMsg}
                  </div>
                )}
                <DialogFooter>
                  <Button type="button" variant="secondary" onClick={closeDialog} disabled={busy}>
                    Отмена
                  </Button>
                  <Button
                    type="submit"
                    variant={dialog.kind === "deactivate" ? "danger" : "primary"}
                    loading={busy}
                    disabled={
                      !confirmSatisfied ||
                      (dialog.kind === "password" && !passOk) ||
                      (dialog.kind === "role" && roleValue === dialog.user.role) ||
                      (stepUpRequired && callerPassword === "")
                    }
                  >
                    {ACTION_META[dialog.kind].confirm}
                  </Button>
                </DialogFooter>
              </form>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Карточка пользователя */}
      <Dialog open={detailId !== null} onOpenChange={(open) => !open && closeDetail()}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{detail?.user?.name || "Карточка пользователя"}</DialogTitle>
            <DialogDescription>{detail ? detail.user.username : "Загрузка данных"}</DialogDescription>
          </DialogHeader>
          {detailPhase === "loading" && (
            <div className="space-y-3">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-24 w-full" />
            </div>
          )}
          {detailPhase === "error" && (
            <ErrorState
              title="Не удалось загрузить карточку"
              description={detailError}
              onRetry={() => detailId && void loadDetail(detailId)}
            />
          )}
          {detailPhase === "ready" && detail && (
            <div className="space-y-4">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
                <dt className="text-xs text-muted-foreground">Email</dt>
                <dd className="text-foreground">{detail.user.username}</dd>
                <dt className="text-xs text-muted-foreground">Роль</dt>
                <dd>
                  <Badge tone={roleTone(detail.user.role)}>{roleLabel(detail.user.role)}</Badge>
                </dd>
                <dt className="text-xs text-muted-foreground">Статус</dt>
                <dd>
                  <Badge tone={detail.user.is_active ? "success" : "danger"}>
                    {detail.user.is_active ? "активен" : "деактивирован"}
                  </Badge>
                </dd>
                <dt className="text-xs text-muted-foreground">2FA</dt>
                <dd>
                  <Badge tone={detail.totp_enabled ? "success" : "neutral"}>
                    {detail.totp_enabled ? "вкл" : "выкл"}
                  </Badge>
                </dd>
                <dt className="text-xs text-muted-foreground">Запросов</dt>
                <dd className="tabular-nums text-foreground">{detail.user.total_requests}</dd>
                <dt className="text-xs text-muted-foreground">Активность</dt>
                <dd className="text-foreground">{fmtDateTime(detail.user.last_activity)}</dd>
                <dt className="text-xs text-muted-foreground">Направления</dt>
                <dd className="col-span-2 flex flex-wrap items-center gap-1">
                  {(detail.directions ?? []).length > 0 ? (
                    detail.directions.map((d) => (
                      <span
                        key={d}
                        className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground"
                      >
                        {d}
                      </span>
                    ))
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </dd>
              </dl>
              <div>
                <h4 className="mb-2 text-sm font-semibold text-foreground">Сессии</h4>
                <TableFrame>
                  <table className="w-full text-sm">
                    <thead className={tableHeadClass}>
                      <tr>
                        <th scope="col" className="px-3 py-2 text-left font-semibold">IP</th>
                        <th scope="col" className="px-3 py-2 text-left font-semibold">
                          User-Agent
                        </th>
                        <th scope="col" className="px-3 py-2 text-left font-semibold">Вход</th>
                        <th scope="col" className="px-3 py-2 text-left font-semibold">
                          Активность
                        </th>
                        <th scope="col" className="px-3 py-2 text-left font-semibold">Статус</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.sessions.length === 0 ? (
                        <tr>
                          <td colSpan={5} className="px-3 py-4 text-center text-muted-foreground">
                            Сессий нет
                          </td>
                        </tr>
                      ) : (
                        detail.sessions.map((s) => (
                          <tr key={s.id} className={tableRowClass}>
                            <td
                              className="px-3 py-1.5 font-mono text-xs"
                              title={s.ip_address || undefined}
                            >
                              {truncate(s.ip_address, 18)}
                            </td>
                            <td
                              className="max-w-[16rem] truncate px-3 py-1.5 text-xs text-muted-foreground"
                              title={s.user_agent || undefined}
                            >
                              {s.user_agent || "-"}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-xs">
                              {fmtDateTime(s.logged_in_at)}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-xs">
                              {fmtDateTime(s.last_activity)}
                            </td>
                            <td className="px-3 py-1.5">
                              <Badge tone={s.is_open ? "success" : "neutral"}>
                                {s.is_open ? "открыта" : "закрыта"}
                              </Badge>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </TableFrame>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
