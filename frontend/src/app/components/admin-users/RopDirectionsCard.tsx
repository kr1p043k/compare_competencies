/**
 * Карточка «Направления РОП»: выбор руководителя ОП и чекбоксы направлений.
 *
 * Перенесено из AdminUsersTab.tsx без изменений разметки, id и классов.
 */
import { Button, Card, CardBody, CardHeader, Field } from "../../../design/primitives";
import { selectClass } from "./constants";
import type { AdminUser, Direction } from "./types";

interface RopDirectionsCardProps {
  directions: Direction[];
  ropList: AdminUser[];
  editUser: string;
  editUserDirs: string[];
  dirBusy: boolean;
  dirMsg: { text: string; ok: boolean } | null;
  onSelectUser: (userId: string) => void;
  onToggleDir: (dirCode: string, checked: boolean) => void;
  onSave: () => void;
}

export function RopDirectionsCard({
  directions,
  ropList,
  editUser,
  editUserDirs,
  dirBusy,
  dirMsg,
  onSelectUser,
  onToggleDir,
  onSave,
}: RopDirectionsCardProps) {
  return (
    <Card>
      <CardHeader
              headingLevel={2}
        title="Направления РОП"
        description="Настройте, какие направления доступны руководителю ОП."
      />
      <CardBody className="space-y-3">
        <Field label="Пользователь" htmlFor="rop-user-select">
          <select
            id="rop-user-select"
            className={`${selectClass} w-full`}
            value={editUser}
            onChange={(e) => onSelectUser(e.target.value)}
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
            <div className="flex items-center gap-3">
              <Button onClick={onSave} loading={dirBusy}>
                Сохранить
              </Button>
              {dirMsg && (
                <p
                  role="status"
                  aria-live="polite"
                  className={dirMsg.ok ? "text-sm text-success" : "text-sm text-destructive"}
                >
                  {dirMsg.text}
                </p>
              )}
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}
