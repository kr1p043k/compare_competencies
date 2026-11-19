/**
 * Шаг «Рекомендации» по дисциплине: форма добавления, фильтры и список.
 *
 * Перенесено из TeacherDashboard.tsx без изменений разметки и классов.
 * Данные и состояние формы приходят пропсами из контейнера.
 */
import type { DisciplineDetail, Recommendation } from "./types";

interface DisciplineRecsSectionProps {
  selected: DisciplineDetail;
  recs: Recommendation[];
  discRecCount: number;
  recsVisible: Recommendation[];
  recFiltersActive: boolean;
  recComp: string;
  setRecComp: (value: string) => void;
  recTypeF: string;
  setRecTypeF: (value: string) => void;
  showAddForm: boolean;
  setShowAddForm: (value: boolean) => void;
  recType: string;
  setRecType: (value: string) => void;
  selectedCompetency: string;
  setSelectedCompetency: (value: string) => void;
  suggestion: string;
  setSuggestion: (value: string) => void;
  recError: string;
  setRecError: (value: string) => void;
  seedMsg: string;
  seedLoading: boolean;
  seedAutoRecs: () => Promise<void>;
  addRec: () => Promise<boolean>;
  deleteRec: (id: number) => Promise<void>;
}

export function DisciplineRecsSection({
  selected,
  recs,
  discRecCount,
  recsVisible,
  recFiltersActive,
  recComp,
  setRecComp,
  recTypeF,
  setRecTypeF,
  showAddForm,
  setShowAddForm,
  recType,
  setRecType,
  selectedCompetency,
  setSelectedCompetency,
  suggestion,
  setSuggestion,
  recError,
  setRecError,
  seedMsg,
  seedLoading,
  seedAutoRecs,
  addRec,
  deleteRec,
}: DisciplineRecsSectionProps) {
  return (
    <div className="mt-1">
      <div className="flex items-center gap-3 mb-3">
        <div className="flex items-center justify-center w-8 h-8 bg-primary rounded-lg">
          <span className="text-primary-foreground text-sm font-bold">!</span>
        </div>
        <h3 className="text-sm font-semibold text-foreground">Рекомендации</h3>
        <span className="text-xs text-subtle-foreground">({recs.filter((r) => r.discipline_id === selected?.name).length})</span>
        {seedMsg && <span className="text-xs text-muted-foreground">{seedMsg}</span>}
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="ml-auto text-xs bg-primary text-primary-foreground px-3 py-1.5 rounded-lg hover:bg-primary transition-colors cursor-pointer border-0"
        >
          {showAddForm ? "Отмена" : "Добавить рекомендацию"}
        </button>
        <button
          onClick={seedAutoRecs}
          disabled={seedLoading}
          title="Заполнить панель топ-рекомендациями из автоанализа (ручные сохранятся)"
          className="text-xs bg-card text-primary-soft-foreground border border-primary/40 px-3 py-1.5 rounded-lg hover:bg-primary-soft transition-colors cursor-pointer disabled:opacity-50"
        >
          {seedLoading ? "Заполнение…" : "Заполнить из анализа"}
        </button>
      </div>
      {showAddForm && (
        <div className="mb-4 p-4 bg-muted rounded-lg border border-border">
          {(recType === "modify" || recType === "remove") && (
            <select
              value={selectedCompetency}
              onChange={(e) => setSelectedCompetency(e.target.value)}
              className="w-full h-9 px-2 mb-2 text-sm bg-input-background border border-border-strong rounded-lg outline-none text-foreground"
            >
              <option value="">-- Select competency --</option>
              {selected?.competencies.map((c) => (
                <option key={c.code} value={c.code}>{c.code}</option>
              ))}
            </select>
          )}
          <textarea
            placeholder={recType === "add" ? "Your recommendation for this discipline..." : "Your recommendation for this competency..."}
            value={suggestion}
            onChange={(e) => setSuggestion(e.target.value)}
            rows={2}
            className="w-full p-2 text-sm border border-border-strong rounded-lg resize-vertical outline-none box-border bg-card text-foreground"
          />
          <div className="flex gap-2 mt-2 items-center">
            <select
              value={recType}
              onChange={(e) => { setRecType(e.target.value); setSelectedCompetency(""); }}
              className="h-9 px-2 text-sm bg-input-background border border-border-strong rounded-lg outline-none text-foreground"
            >
              <option value="modify">Изменить</option>
              <option value="add">Добавить</option>
              <option value="remove">Удалить</option>
            </select>
            <button
              onClick={async () => {
                setRecError("");
                const ok = await addRec();
                if (ok) {
                  setShowAddForm(false);
                  setSelectedCompetency("");
                }
              }}
              className="h-9 px-4 text-sm bg-primary text-primary-foreground border-0 rounded-lg cursor-pointer hover:bg-primary transition-colors"
            >
              Отправить
            </button>
          </div>
          {recError && (
            <div className="mt-2 text-xs text-destructive">
              Не удалось сохранить: {recError}
            </div>
          )}
        </div>
      )}
      {discRecCount > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <select
            value={recComp}
            onChange={(e) => setRecComp(e.target.value)}
            aria-label="Фильтр по компетенции"
            title="Показать рекомендации только по одной компетенции"
            className="h-8 px-2 rounded-md text-xs bg-input-background border border-border-strong text-foreground"
          >
            <option value="">Все компетенции</option>
            {selected.competencies.map((c) => (
              <option key={c.code} value={c.code}>{c.code}{c.name ? ` — ${c.name}` : ""}</option>
            ))}
          </select>
          {([
            ["", "Все"],
            ["modify", "Изменить"],
            ["add", "Добавить"],
            ["remove", "Удалить"],
          ] as const).map(([id, label]) => (
            <button
              key={id || "all"}
              onClick={() => setRecTypeF(id)}
              aria-pressed={recTypeF === id}
              title={id ? `Только рекомендации типа «${label.toLowerCase()}»` : "Все типы рекомендаций"}
              className={
                "text-xs px-2 py-0.5 rounded border cursor-pointer " +
                (recTypeF === id
                  ? "border-primary/60 bg-primary-soft text-primary-soft-foreground"
                  : "border-border-strong bg-card text-muted-foreground")
              }
            >
              {label}
            </button>
          ))}
          <span className="text-[11px] text-subtle-foreground ml-auto whitespace-nowrap">
            {recsVisible.length} из {discRecCount}
          </span>
          {recFiltersActive && (
            <button
              onClick={() => { setRecComp(""); setRecTypeF(""); }}
              className="text-[11px] text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer underline whitespace-nowrap"
            >
              Сбросить
            </button>
          )}
        </div>
      )}
      {discRecCount === 0 ? (
        <div className="text-xs text-subtle-foreground bg-muted rounded-lg p-4 text-center">
          Для этой дисциплины рекомендаций пока нет
        </div>
      ) : recsVisible.length === 0 ? (
        <div className="text-xs text-subtle-foreground bg-muted rounded-lg p-4 text-center">
          По фильтру ничего не найдено —{" "}
          <button
            onClick={() => { setRecComp(""); setRecTypeF(""); }}
            className="text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer underline"
          >
            сбросить
          </button>
        </div>
      ) : (
        recsVisible.map((r, i) => (
          <div
            key={r.id ?? i}
            className="mb-2 rounded-lg border border-primary/30 bg-primary-soft/60 p-3 text-sm "
          >
            <div className="text-subtle-foreground mb-1 text-xs">
              [{r.suggestion_type}] {r.competency_id || "—"}
            </div>
            <div className="text-foreground">{r.suggestion}</div>
            <button
              onClick={() => { if (window.confirm("Удалить рекомендацию безвозвратно?")) deleteRec(r.id); }}
              className="mt-2 text-xs text-destructive border border-destructive/60 rounded px-2 py-0.5 hover:bg-destructive/10 transition-colors bg-transparent cursor-pointer"
            >
              Удалить
            </button>
          </div>
        ))
      )}
    </div>
  );
}
