import { Cpu, Loader2, Plus, X } from "lucide-react";
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
import type { SelfData } from "../types";

type MyTechnologiesCardProps = {
  self: SelfData | null;
  selfLoading: boolean;
  techList: string[];
  techSaving: boolean;
  techInput: string;
  onTechInputChange: (value: string) => void;
  onAddTech: (raw?: string) => void;
  onRemoveTech: (text: string) => void;
  techSuggestions: string[];
  suggestPool: string[];
  techError: string | null;
};

export function MyTechnologiesCard({
  self,
  selfLoading,
  techList,
  techSaving,
  techInput,
  onTechInputChange,
  onAddTech,
  onRemoveTech,
  techSuggestions,
  suggestPool,
  techError,
}: MyTechnologiesCardProps) {
  return (
    <Card id="my-technologies">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <Cpu className="size-5 text-success" />
            Мои технологии{self && !selfLoading ? " (" + techList.length + ")" : ""}
          </span>
        }
        description="Стек, с которым вы работаете: добавляйте новое и убирайте устаревшее"
      />
      <CardBody className="space-y-3">
        {selfLoading && (
          <div className="flex flex-wrap gap-1.5" aria-label="Загрузка технологий">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-6 w-20" />
            ))}
          </div>
        )}
        {self && !selfLoading && (
          <>
            {techList.length === 0 ? (
              <EmptyState title="Пока пусто" description="Добавьте первую технологию ниже. Подсказки берутся из списка technologies_suggest." />
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {techList.map((t) => (
                  <Badge key={t} tone="neutral" className="text-xs gap-1">
                    {t}
                    <button
                      onClick={() => onRemoveTech(t)}
                      disabled={techSaving}
                      className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      aria-label={"Убрать " + t}
                    >
                      <X className="size-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                value={techInput}
                onChange={(e) => onTechInputChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && techInput.trim()) onAddTech();
                }}
                placeholder="Например: Docker. Enter - добавить"
                aria-label="Новая технология"
                list="my-tech-suggest"
                className={inputClass}
              />
              <datalist id="my-tech-suggest">
                {suggestPool.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
              <Button
                variant="success"
                onClick={() => onAddTech()}
                disabled={techSaving || !techInput.trim()}
                className="shrink-0 cursor-pointer"
                aria-label="Добавить технологию"
              >
                {techSaving ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              </Button>
            </div>
            {techSuggestions.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {techSuggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => onAddTech(s)}
                    disabled={techSaving}
                    className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border bg-card px-2 py-0.5 text-xs text-foreground transition-colors duration-200 hover:border-success/60 hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                    aria-label={"Добавить " + s}
                  >
                    <Plus className="size-3" />
                    {s}
                  </button>
                ))}
              </div>
            )}
            {techError && <p className="text-sm text-destructive">{techError}</p>}
          </>
        )}
      </CardBody>
    </Card>
  );
}
