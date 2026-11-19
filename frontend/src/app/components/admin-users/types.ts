/**
 * Типы вкладки «Пользователи» админ-консоли.
 *
 * Вынесены из AdminUsersTab.tsx без изменения формы данных: те же поля и
 * те же серверные ответы (GET /api/admin/users, GET /api/admin/users/{id}).
 */
import type { Role } from "../adminRoles";

export interface Direction {
  dir_code: string;
  name: string;
}

export interface AdminUser {
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

export interface UsersPage {
  users: AdminUser[];
  total: number;
  limit: number;
  offset: number;
}

export interface UserSession {
  id: string;
  ip_address: string;
  user_agent: string;
  logged_in_at: string | number | null;
  last_activity: string | number | null;
  is_open: boolean;
}

export interface UserDetail {
  user: AdminUser;
  directions: string[];
  sessions: UserSession[];
  totp_enabled: boolean;
}

export type ActionKind = "deactivate" | "password" | "revoke" | "twofa" | "role";

export interface ActionDialog {
  kind: ActionKind;
  user: AdminUser;
}
