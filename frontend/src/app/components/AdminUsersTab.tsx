/**
 * Вкладка «Пользователи» админ-консоли.
 *
 * Серверная пагинация и фильтры (GET /api/admin/users?q=&role=&is_active=&limit=&offset=),
 * карточка пользователя с сессиями (GET /api/admin/users/{id}), действия PATCH/POST,
 * создание пользователя и редактор направлений РОП. Все необратимые действия требуют
 * ввода подтверждения (адрес пользователя или слово УДАЛИТЬ) в диалоге.
 * Действия над другим администратором дополнительно требуют пароль текущего
 * администратора (caller_password), иначе сервер отвечает 403.
 *
 * Разметка разнесена по ./admin-users (таблица, фильтры, карточки, диалоги);
 * состояние, загрузка данных и проверки подтверждений остаются здесь.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";

import { request } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
} from "../../design/primitives";
import { type Role, roleLabel } from "./adminRoles";
import { ActionDialogView } from "./admin-users/ActionDialogView";
import { CreateUserCard } from "./admin-users/CreateUserCard";
import { DEACTIVATION_WORD, STEP_UP_KINDS } from "./admin-users/constants";
import { RopDirectionsCard } from "./admin-users/RopDirectionsCard";
import type {
  ActionDialog,
  ActionKind,
  AdminUser,
  Direction,
  UserDetail,
  UsersPage,
} from "./admin-users/types";
import { UserDetailDialog } from "./admin-users/UserDetailDialog";
import { UsersFilters } from "./admin-users/UsersFilters";
import { UsersTable } from "./admin-users/UsersTable";

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

  /* --------------------------------------------------------- фильтры и выборки */

  const changeRoleFilter = (value: "all" | Role) => {
    setRoleFilter(value);
    setOffset(0);
  };

  const changeStatusFilter = (value: "all" | "active" | "inactive") => {
    setStatusFilter(value);
    setOffset(0);
  };

  const changeLimit = (value: number) => {
    setLimit(value);
    setOffset(0);
  };

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

  const retryDetail = () => {
    if (detailId) void loadDetail(detailId);
  };

  /* ------------------------------------------------------------ создание, РОП */

  const changeNewUserRole = (role: Role) => {
    setNewUserRole(role);
    setNewUserDirs([]);
  };

  const toggleNewUserDir = (dirCode: string, checked: boolean) => {
    setNewUserDirs((prev) =>
      checked ? [...prev, dirCode] : prev.filter((x) => x !== dirCode),
    );
  };

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

  const selectRopUser = (userId: string) => {
    const u = ropList.find((x) => x.id === userId);
    setEditUser(userId);
    setEditUserDirs(u?.directions ?? []);
    setDirMsg(null);
  };

  const toggleEditUserDir = (dirCode: string, checked: boolean) => {
    setEditUserDirs((prev) =>
      checked ? [...prev, dirCode] : prev.filter((x) => x !== dirCode),
    );
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
              ? "rounded-lg border border-success/30 bg-success-soft px-3 py-2 text-sm text-success"
              : "rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
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
          <UsersFilters
            search={search}
            roleFilter={roleFilter}
            statusFilter={statusFilter}
            limit={limit}
            onSearchChange={setSearch}
            onRoleChange={changeRoleFilter}
            onStatusChange={changeStatusFilter}
            onLimitChange={changeLimit}
          />

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
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
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
            <UsersTable
              showSkeleton={showSkeleton}
              rows={rows}
              onOpenDetail={openDetail}
              onOpenAction={openAction}
              onActivate={activateUser}
            />
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
      <CreateUserCard
        directions={directions}
        email={newUserEmail}
        password={newUserPass}
        name={newUserName}
        role={newUserRole}
        dirs={newUserDirs}
        busy={createBusy}
        result={userCreated}
        onEmailChange={setNewUserEmail}
        onPasswordChange={setNewUserPass}
        onNameChange={setNewUserName}
        onRoleChange={changeNewUserRole}
        onToggleDir={toggleNewUserDir}
        onCreate={() => void createUser()}
      />

      {/* Редактор направлений РОП */}
      <RopDirectionsCard
        directions={directions}
        ropList={ropList}
        editUser={editUser}
        editUserDirs={editUserDirs}
        dirBusy={dirBusy}
        dirMsg={dirMsg}
        onSelectUser={selectRopUser}
        onToggleDir={toggleEditUserDir}
        onSave={() => void saveDirections()}
      />

      {/* Диалог подтверждения действия */}
      <ActionDialogView
        dialog={dialog}
        confirmText={confirmText}
        newPass1={newPass1}
        newPass2={newPass2}
        callerPassword={callerPassword}
        roleValue={roleValue}
        needWord={needWord}
        confirmSatisfied={confirmSatisfied}
        stepUpRequired={stepUpRequired}
        passOk={passOk}
        busy={busy}
        dialogMsg={dialogMsg}
        dialogErr={dialogErr}
        onConfirmTextChange={setConfirmText}
        onNewPass1Change={setNewPass1}
        onNewPass2Change={setNewPass2}
        onCallerPasswordChange={setCallerPassword}
        onRoleValueChange={setRoleValue}
        onSubmit={() => void submitDialog()}
        onClose={closeDialog}
      />

      {/* Карточка пользователя */}
      <UserDetailDialog
        detailId={detailId}
        detail={detail}
        detailPhase={detailPhase}
        detailError={detailError}
        onRetry={retryDetail}
        onClose={closeDetail}
      />
    </div>
  );
}
