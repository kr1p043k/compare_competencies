/**
 * Таблица пользователей: шапка, скелетон загрузки и строки UserRow.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки и классов.
 */
import { Skeleton, TableFrame, tableHeadClass, tableRowClass } from "../../../design/primitives";
import { SKELETON_ROWS } from "./constants";
import type { ActionKind, AdminUser } from "./types";
import { UserRow } from "./UserRow";

interface UsersTableProps {
  showSkeleton: boolean;
  rows: AdminUser[];
  onOpenDetail: (id: string) => void;
  onOpenAction: (kind: ActionKind, user: AdminUser) => void;
  onActivate: (user: AdminUser) => void;
}

export function UsersTable({
  showSkeleton,
  rows,
  onOpenDetail,
  onOpenAction,
  onActivate,
}: UsersTableProps) {
  return (
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
                <UserRow
                  key={u.id}
                  user={u}
                  onOpenDetail={onOpenDetail}
                  onOpenAction={onOpenAction}
                  onActivate={onActivate}
                />
              ))}
        </tbody>
      </table>
    </TableFrame>
  );
}
