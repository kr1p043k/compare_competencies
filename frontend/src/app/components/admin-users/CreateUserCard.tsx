/**
 * Карточка «Создать пользователя»: почта, пароль, ФИО, роль и направления РОП.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки, id и классов.
 * Состояние формы и запрос остаются в AdminUsersTab, здесь только презентация
 * и колбэк onCreate.
 */
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Field,
  inputClass,
} from "../../../design/primitives";
import { asRole, type Role } from "../adminRoles";
import { selectClass } from "./constants";
import type { Direction } from "./types";

interface CreateUserCardProps {
  directions: Direction[];
  email: string;
  password: string;
  name: string;
  role: Role;
  dirs: string[];
  busy: boolean;
  result: { text: string; ok: boolean } | null;
  onEmailChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onNameChange: (value: string) => void;
  onRoleChange: (role: Role) => void;
  onToggleDir: (dirCode: string, checked: boolean) => void;
  onCreate: () => void;
}

export function CreateUserCard({
  directions,
  email,
  password,
  name,
  role,
  dirs,
  busy,
  result,
  onEmailChange,
  onPasswordChange,
  onNameChange,
  onRoleChange,
  onToggleDir,
  onCreate,
}: CreateUserCardProps) {
  return (
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
            onCreate();
          }}
        >
          <Field label="Электронная почта" htmlFor="new-user-email">
            <input
              id="new-user-email"
              type="email"
              className={inputClass}
              value={email}
              onChange={(e) => onEmailChange(e.target.value)}
              autoComplete="off"
            />
          </Field>
          <Field label="Пароль" htmlFor="new-user-pass" hint="Минимум 10 символов">
            <input
              id="new-user-pass"
              type="password"
              className={inputClass}
              value={password}
              onChange={(e) => onPasswordChange(e.target.value)}
              autoComplete="new-password"
            />
          </Field>
          <Field label="ФИО" htmlFor="new-user-name">
            <input
              id="new-user-name"
              className={inputClass}
              value={name}
              onChange={(e) => onNameChange(e.target.value)}
              autoComplete="off"
            />
          </Field>
          <Field label="Роль" htmlFor="new-user-role">
            <select
              id="new-user-role"
              className={`${selectClass} w-full`}
              value={role}
              onChange={(e) => onRoleChange(asRole(e.target.value))}
            >
              <option value="teacher">Преподаватель</option>
              <option value="admin">Администратор</option>
              <option value="rop">Руководитель ОП</option>
            </select>
          </Field>
          {role === "rop" && (
            <Field label="Направления" htmlFor="new-user-dirs">
              <div
                id="new-user-dirs"
                className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-border p-3"
              >
                {directions.map((d) => (
                  <label key={d.dir_code} className="flex cursor-pointer items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={dirs.includes(d.dir_code)}
                      onChange={(e) => onToggleDir(d.dir_code, e.target.checked)}
                      className="accent-primary"
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
            <Button type="submit" loading={busy}>
              Создать
            </Button>
            {result && (
              <p
                role="status"
                aria-live="polite"
                className={result.ok ? "text-sm text-success" : "text-sm text-destructive"}
              >
                {result.text}
              </p>
            )}
          </div>
        </form>
      </CardBody>
    </Card>
  );
}
