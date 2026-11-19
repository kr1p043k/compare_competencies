/**
 * Диалог подтверждения действия над пользователем: роль, пароль, отзыв сессий,
 * деактивация, сброс 2FA. Включает поле step-up пароля администратора.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки, id и классов.
 * Проверки (needWord, confirmSatisfied, stepUpRequired, passOk) считаются
 * в AdminUsersTab и приходят пропсами; запрос выполняет onSubmit.
 */
import { Button, Field, inputClass } from "../../../design/primitives";
import { asRole, ROLES, ROLE_LABEL, type Role } from "../adminRoles";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { ACTION_META, selectClass } from "./constants";
import type { ActionDialog } from "./types";

interface ActionDialogViewProps {
  dialog: ActionDialog | null;
  confirmText: string;
  newPass1: string;
  newPass2: string;
  callerPassword: string;
  roleValue: Role;
  needWord: boolean;
  confirmSatisfied: boolean;
  stepUpRequired: boolean;
  passOk: boolean;
  busy: boolean;
  dialogMsg: string;
  dialogErr: string;
  onConfirmTextChange: (value: string) => void;
  onNewPass1Change: (value: string) => void;
  onNewPass2Change: (value: string) => void;
  onCallerPasswordChange: (value: string) => void;
  onRoleValueChange: (role: Role) => void;
  onSubmit: () => void;
  onClose: () => void;
}

export function ActionDialogView({
  dialog,
  confirmText,
  newPass1,
  newPass2,
  callerPassword,
  roleValue,
  needWord,
  confirmSatisfied,
  stepUpRequired,
  passOk,
  busy,
  dialogMsg,
  dialogErr,
  onConfirmTextChange,
  onNewPass1Change,
  onNewPass2Change,
  onCallerPasswordChange,
  onRoleValueChange,
  onSubmit,
  onClose,
}: ActionDialogViewProps) {
  return (
    <Dialog open={dialog !== null} onOpenChange={(open) => !open && onClose()}>
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
                onSubmit();
              }}
            >
              {dialog.kind === "role" && (
                <Field label="Роль" htmlFor="dialog-role">
                  <select
                    id="dialog-role"
                    className={`${selectClass} w-full`}
                    value={roleValue}
                    onChange={(e) => onRoleValueChange(asRole(e.target.value))}
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
                      onChange={(e) => onNewPass1Change(e.target.value)}
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
                      onChange={(e) => onNewPass2Change(e.target.value)}
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
                    onChange={(e) => onConfirmTextChange(e.target.value)}
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
                    onChange={(e) => onCallerPasswordChange(e.target.value)}
                    autoComplete="current-password"
                    required
                    aria-invalid={callerPassword === "" || undefined}
                  />
                </Field>
              )}
              {dialogErr && (
                <div
                  role="alert"
                  className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                >
                  {dialogErr}
                </div>
              )}
              {dialogMsg && (
                <div
                  role="status"
                  aria-live="polite"
                  className="rounded-lg border border-success/30 bg-success-soft px-3 py-2 text-sm text-success"
                >
                  {dialogMsg}
                </div>
              )}
              <DialogFooter>
                <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
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
  );
}
