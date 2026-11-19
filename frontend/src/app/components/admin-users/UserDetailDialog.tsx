/**
 * Карточка пользователя: реквизиты и таблица сессий.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки и классов.
 * Загрузка и состояния (loading/error/ready) живут в AdminUsersTab.
 */
import {
  Badge,
  ErrorState,
  Skeleton,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../../design/primitives";
import { fmtDateTime, roleLabel, roleTone } from "../adminRoles";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { truncate } from "./helpers";
import type { UserDetail } from "./types";

interface UserDetailDialogProps {
  detailId: string | null;
  detail: UserDetail | null;
  detailPhase: "idle" | "loading" | "error" | "ready";
  detailError: string;
  onRetry: () => void;
  onClose: () => void;
}

export function UserDetailDialog({
  detailId,
  detail,
  detailPhase,
  detailError,
  onRetry,
  onClose,
}: UserDetailDialogProps) {
  return (
    <Dialog open={detailId !== null} onOpenChange={(open) => !open && onClose()}>
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
            onRetry={onRetry}
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
  );
}
