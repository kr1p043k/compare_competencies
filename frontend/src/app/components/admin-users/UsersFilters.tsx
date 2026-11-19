/**
 * Блок фильтров вкладки «Пользователи»: поиск, роль, статус, размер страницы.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки, id и классов.
 * Значения и колбэки приходят пропсами - фильтрация считается на сервере.
 */
import { Field, inputClass } from "../../../design/primitives";
import type { Role } from "../adminRoles";
import { selectClass } from "./constants";

interface UsersFiltersProps {
  search: string;
  roleFilter: "all" | Role;
  statusFilter: "all" | "active" | "inactive";
  limit: number;
  onSearchChange: (value: string) => void;
  onRoleChange: (value: "all" | Role) => void;
  onStatusChange: (value: "all" | "active" | "inactive") => void;
  onLimitChange: (value: number) => void;
}

export function UsersFilters({
  search,
  roleFilter,
  statusFilter,
  limit,
  onSearchChange,
  onRoleChange,
  onStatusChange,
  onLimitChange,
}: UsersFiltersProps) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <Field label="Поиск" htmlFor="users-search" className="w-64 max-w-full">
        <input
          id="users-search"
          type="search"
          className={inputClass}
          placeholder="Имя или email"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </Field>
      <Field label="Роль" htmlFor="users-role">
        <select
          id="users-role"
          className={selectClass}
          value={roleFilter}
          onChange={(e) => onRoleChange(e.target.value as "all" | Role)}
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
          onChange={(e) => onStatusChange(e.target.value as "all" | "active" | "inactive")}
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
          onChange={(e) => onLimitChange(Number(e.target.value) || 25)}
        >
          <option value="25">25</option>
          <option value="50">50</option>
          <option value="100">100</option>
        </select>
      </Field>
    </div>
  );
}
