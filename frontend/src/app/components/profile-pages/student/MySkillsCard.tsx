import { GraduationCap, Loader2, Plus, X } from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  Skeleton,
  inputClass,
} from "../../../../design";
import { norm } from "../helpers";
import type { SelfData } from "../types";

type MySkillsCardProps = {
  self: SelfData | null;
  selfLoading: boolean;
  selfError: string | null;
  skillsCount: number;
  owned: Set<string>;
  saving: boolean;
  newSkill: string;
  onNewSkillChange: (value: string) => void;
  onAddSkill: () => void;
  onRemoveSkill: (text: string) => void;
  opError: string | null;
};

export function MySkillsCard({
  self,
  selfLoading,
  selfError,
  skillsCount,
  owned,
  saving,
  newSkill,
  onNewSkillChange,
  onAddSkill,
  onRemoveSkill,
  opError,
}: MySkillsCardProps) {
  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <GraduationCap className="size-5 text-success" />
            Мои навыки{self ? ` (${skillsCount})` : ""}
          </span>
        }
        description="Свои навыки хранятся в вашем профиле - добавляйте и удаляйте"
      />
      <CardBody className="space-y-3">
        {selfLoading && (
          <div className="flex flex-wrap gap-1.5" aria-label="Загрузка навыков">
            {[0, 1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-6 w-20" />
            ))}
          </div>
        )}
        {selfError && !selfLoading && (
          <p className="text-sm text-destructive">{selfError}</p>
        )}
        {self && !selfLoading && (
          <>
            {skillsCount === 0 ? (
              <EmptyState title="Навыков пока нет" description="Добавьте первый ниже" />
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {self.skills.map((s) => {
                  const mine = owned.has(norm(s));
                  return (
                    <Badge
                      key={s}
                      tone={mine ? "primary" : "neutral"}
                      className="text-xs gap-1"
                      title={mine ? "Добавлен вами - можно удалить" : "Из программы - удалить нельзя"}
                    >
                      {s}
                      {mine && (
                        <button
                          onClick={() => onRemoveSkill(s)}
                          disabled={saving}
                          className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                          aria-label={`Удалить ${s}`}
                        >
                          <X className="size-3" />
                        </button>
                      )}
                    </Badge>
                  );
                })}
              </div>
            )}
            <div className="flex gap-2">
              <input
                value={newSkill}
                onChange={(e) => onNewSkillChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newSkill.trim()) onAddSkill();
                }}
                placeholder="Новый навык, Enter - добавить"
                aria-label="Новый навык"
                className={inputClass}
              />
              <Button
                variant="success"
                onClick={() => onAddSkill()}
                disabled={saving || !newSkill.trim()}
                className="shrink-0 cursor-pointer"
                aria-label="Добавить навык"
              >
                {saving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
            {opError && <p className="text-sm text-destructive">{opError}</p>}
          </>
        )}
      </CardBody>
    </Card>
  );
}
