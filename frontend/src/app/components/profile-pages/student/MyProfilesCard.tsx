import { CheckCircle2, Layers, Loader2, Plus, Search, X } from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Skeleton,
  inputClass,
} from "../../../../design";
import { Label } from "../../ui/label";
import { Textarea } from "../../ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../ui/select";
import { cn } from "../../ui/utils";
import { NO_BASE } from "../helpers";
import type { CreatedCustomProfile, CustomOptionCompetency, NewCompetencyDraft } from "../types";

type MyProfilesCardProps = {
  optionsLoading: boolean;
  optionsError: string | null;
  createdProfiles: CreatedCustomProfile[];
  cpName: string;
  onCpNameChange: (value: string) => void;
  cpNameValid: boolean;
  cpLevel: string;
  onCpLevelChange: (value: string) => void;
  cpBase: string;
  onCpBaseChange: (value: string) => void;
  baseProfiles: string[];
  compCodes: string[];
  compSearch: string;
  onCompSearchChange: (value: string) => void;
  filteredCustomComps: CustomOptionCompetency[];
  onToggleCompCode: (code: string) => void;
  draftCode: string;
  onDraftCodeChange: (value: string) => void;
  draftTitle: string;
  onDraftTitleChange: (value: string) => void;
  draftKnowledge: string;
  onDraftKnowledgeChange: (value: string) => void;
  draftAbilities: string;
  onDraftAbilitiesChange: (value: string) => void;
  draftSkills: string;
  onDraftSkillsChange: (value: string) => void;
  onAddDraft: () => void;
  onRemoveDraft: (code: string) => void;
  newComps: NewCompetencyDraft[];
  cpTech: string;
  onCpTechChange: (value: string) => void;
  triedCreate: boolean;
  totalComps: number;
  createError: string | null;
  creating: boolean;
  onCreateProfile: () => void;
};

export function MyProfilesCard({
  optionsLoading,
  optionsError,
  createdProfiles,
  cpName,
  onCpNameChange,
  cpNameValid,
  cpLevel,
  onCpLevelChange,
  cpBase,
  onCpBaseChange,
  baseProfiles,
  compCodes,
  compSearch,
  onCompSearchChange,
  filteredCustomComps,
  onToggleCompCode,
  draftCode,
  onDraftCodeChange,
  draftTitle,
  onDraftTitleChange,
  draftKnowledge,
  onDraftKnowledgeChange,
  draftAbilities,
  onDraftAbilitiesChange,
  draftSkills,
  onDraftSkillsChange,
  onAddDraft,
  onRemoveDraft,
  newComps,
  cpTech,
  onCpTechChange,
  triedCreate,
  totalComps,
  createError,
  creating,
  onCreateProfile,
}: MyProfilesCardProps) {
  return (
    <Card id="my-profiles">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <Layers className="size-5 text-primary" />
            Мои профили{createdProfiles.length > 0 ? " (" + createdProfiles.length + ")" : ""}
          </span>
        }
        description="Свои профили компетенций: соберите из готовых или опишите новые"
      />
      <CardBody className="space-y-4">
        {optionsLoading && (
          <div className="space-y-2" aria-label="Загрузка опций профилей">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-9" />
          </div>
        )}
        {optionsError && <ErrorState description={optionsError} />}
        {createdProfiles.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium text-foreground">
              Создано в этой сессии ({createdProfiles.length})
            </div>
            {createdProfiles.map((cp) => (
              <div
                key={cp.name}
                className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-success/30 bg-success-soft px-3 py-2 transition-colors duration-200"
              >
                <span className="flex items-center gap-2 text-sm font-medium text-success">
                  <CheckCircle2 className="size-4 shrink-0" />
                  {cp.name}
                </span>
                <span className="flex flex-wrap gap-1.5">
                  <Badge tone="neutral" className="text-xs">{cp.target_level}</Badge>
                  <Badge tone="neutral" className="text-xs">компетенций: {cp.competencies}</Badge>
                  {cp.created_new > 0 && (
                    <Badge tone="neutral" className="text-xs">новых: {cp.created_new}</Badge>
                  )}
                  {cp.technologies > 0 && (
                    <Badge tone="neutral" className="text-xs">технологий: {cp.technologies}</Badge>
                  )}
                </span>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">
              Сервер отдаёт только общий список профилей, поэтому здесь показаны профили, созданные в этой сессии.
            </p>
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cp-name">Название профиля</Label>
            <input
              id="cp-name"
              value={cpName}
              onChange={(e) => onCpNameChange(e.target.value)}
              placeholder="my_backend"
              aria-label="Название профиля"
              className={inputClass}
            />
            <p className="text-xs text-muted-foreground">Латиница, цифры и подчёркивание, 2-32 символа</p>
          </div>
          <div className="space-y-1.5">
            <Label>Целевой уровень</Label>
            <Select value={cpLevel} onValueChange={onCpLevelChange}>
              <SelectTrigger className="h-10" aria-label="Целевой уровень">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="junior">Junior</SelectItem>
                <SelectItem value="middle">Middle</SelectItem>
                <SelectItem value="senior">Senior</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-1.5">
          <Label>База</Label>
          <Select value={cpBase} onValueChange={onCpBaseChange}>
            <SelectTrigger className="h-10 w-full sm:w-64" aria-label="Базовый профиль">
              <SelectValue placeholder="Без базы" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NO_BASE}>Без базы</SelectItem>
              {baseProfiles.map((b) => (
                <SelectItem key={b} value={b}>
                  {b}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2 rounded-lg border border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <Label>Компетенции</Label>
            <Badge tone="neutral" className="text-xs">выбрано: {compCodes.length}</Badge>
          </div>

          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <input
              value={compSearch}
              onChange={(e) => onCompSearchChange(e.target.value)}
              placeholder="Поиск по коду или названию"
              aria-label="Поиск компетенций"
              className={cn(inputClass, "h-9 pl-8")}
            />
          </div>
          {filteredCustomComps.length === 0 ? (
            <EmptyState title="Нет компетенций для выбора" description="Проверьте поиск или добавьте новую ниже" />
          ) : (
            <ul className="max-h-56 space-y-1 overflow-y-auto pr-1">
              {filteredCustomComps.map((c) => {
                const checked = compCodes.includes(c.code);
                return (
                  <li key={c.code}>
                    <label className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 transition-colors duration-200 hover:bg-muted">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => onToggleCompCode(c.code)}
                        className="mt-1 size-4 shrink-0 cursor-pointer accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
                        aria-label={c.code}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-medium text-foreground">{c.code}</span>
                        {c.title ? (
                          <span className="block truncate text-xs text-muted-foreground">{c.title}</span>
                        ) : null}
                      </span>
                      {typeof c.skills_count === "number" ? (
                        <Badge tone="neutral" className="shrink-0 text-xs">
                          {c.skills_count}
                        </Badge>
                      ) : null}
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <div className="space-y-2 rounded-lg border border-border p-3">
          <Label>Новая компетенция</Label>

          <div className="grid gap-2 sm:grid-cols-2">
            <input
              value={draftCode}
              onChange={(e) => onDraftCodeChange(e.target.value)}
              placeholder="Код, например ПК-5"
              aria-label="Код компетенции"
              className={inputClass}
            />
            <input
              value={draftTitle}
              onChange={(e) => onDraftTitleChange(e.target.value)}
              placeholder="Название (необязательно)"
              aria-label="Название компетенции"
              className={inputClass}
            />
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            <Textarea
              value={draftKnowledge}
              onChange={(e) => onDraftKnowledgeChange(e.target.value)}
              placeholder="Знания - по одному на строку"
              aria-label="Знания"
              rows={3}
              className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            />
            <Textarea
              value={draftAbilities}
              onChange={(e) => onDraftAbilitiesChange(e.target.value)}
              placeholder="Умения - по одному на строку"
              aria-label="Умения"
              rows={3}
              className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            />
            <Textarea
              value={draftSkills}
              onChange={(e) => onDraftSkillsChange(e.target.value)}
              placeholder="Навыки - по одному на строку"
              aria-label="Навыки"
              rows={3}
              className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            />
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={onAddDraft}
            className="cursor-pointer"
            iconLeft={<Plus className="size-3.5" />}
          >
            Добавить компетенцию
          </Button>
          {newComps.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {newComps.map((d) => (
                <Badge key={d.code} tone="primary" className="text-xs gap-1">
                  {d.code}
                  <button
                    onClick={() => onRemoveDraft(d.code)}
                    className="ml-1 cursor-pointer rounded p-0.5 transition-colors duration-200 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                    aria-label={"Убрать " + d.code}
                  >
                    <X className="size-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="cp-tech">Технологии (через запятую)</Label>
          <input
            id="cp-tech"
            value={cpTech}
            onChange={(e) => onCpTechChange(e.target.value)}
            placeholder="Python, Docker, PostgreSQL"
            aria-label="Технологии профиля"
            className={inputClass}
          />
        </div>
        {!cpNameValid && cpName.length > 0 && (
          <p className="text-sm text-destructive">Название: латиница, цифры и подчёркивание, 2-32 символа</p>
        )}
        {triedCreate && totalComps < 1 && (
          <p className="text-sm text-warning">Выберите хотя бы одну компетенцию или добавьте новую</p>
        )}
        {createError && <p className="text-sm text-destructive">{createError}</p>}
        <Button
          variant="primary"
          onClick={() => onCreateProfile()}
          disabled={creating}
          className="cursor-pointer"
        >
          {creating ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Plus className="mr-2 size-4" />}
          Создать профиль
        </Button>
      </CardBody>
    </Card>
  );
}
