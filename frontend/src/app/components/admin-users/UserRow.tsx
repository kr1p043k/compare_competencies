/**
 * Строка таблицы пользователей: имя, роль, направления, сессии, 2FA, действия.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки и классов:
 * обработчики приходят пропсами, локального состояния нет.
 */
import {
  Ellipsis,
  IdCard,
  KeyRound,
  LogOut,
  RotateCcw,
  ShieldCheck,
  ShieldOff,
  UserCog,
} from "lucide-react";

import { Badge, tableRowClass } from "../../../design/primitives";
import { fmtDateTime, roleLabel, roleTone } from "../adminRoles";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import type { ActionKind, AdminUser } from "./types";

interface UserRowProps {
  user: AdminUser;
  onOpenDetail: (id: string) => void;
  onOpenAction: (kind: ActionKind, user: AdminUser) => void;
  onActivate: (user: AdminUser) => void;
}

export function UserRow({ user: u, onOpenDetail, onOpenAction, onActivate }: UserRowProps) {
  return (
    <tr className={tableRowClass}>
      <td className="px-3 py-2 font-medium text-foreground">
        <span className={u.is_active ? "" : "opacity-60"}>{u.name || "-"}</span>
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
              <span className="text-xs text-muted-foreground">+{u.directions.length - 3}</span>
            )}
          </span>
        ) : (
          <span className="text-muted-foreground">-</span>
        )}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-foreground">{u.open_sessions}</td>
      <td className="px-3 py-2">
        <Badge tone={u.totp_enabled ? "success" : "neutral"}>{u.totp_enabled ? "вкл" : "выкл"}</Badge>
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-xs text-muted-foreground">
        {fmtDateTime(u.last_activity)}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{u.total_requests}</td>
      <td className="px-3 py-2 text-right">
        <DropdownMenu>
          <DropdownMenuTrigger
            aria-label={`Действия: ${u.name} (${u.username})`}
            className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Ellipsis className="size-4" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-52">
            <DropdownMenuItem onSelect={() => onOpenDetail(u.id)}>
              <IdCard className="size-4" /> Карточка
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onOpenAction("role", u)}>
              <UserCog className="size-4" /> Сменить роль
            </DropdownMenuItem>
            {u.is_active ? (
              <DropdownMenuItem
                variant="destructive"
                onSelect={() => onOpenAction("deactivate", u)}
              >
                <ShieldOff className="size-4" /> Деактивировать
              </DropdownMenuItem>
            ) : (
              <DropdownMenuItem onSelect={() => void onActivate(u)}>
                <ShieldCheck className="size-4" /> Активировать
              </DropdownMenuItem>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => onOpenAction("password", u)}>
              <KeyRound className="size-4" /> Сбросить пароль
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onOpenAction("revoke", u)}>
              <LogOut className="size-4" /> Отозвать сессии
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onOpenAction("twofa", u)} disabled={!u.totp_enabled}>
              <RotateCcw className="size-4" /> Сбросить 2FA
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </td>
    </tr>
  );
}
