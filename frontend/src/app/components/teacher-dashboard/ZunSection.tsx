/**
 * Шаг «Компетенции и ЗУН»: создание компетенции, фильтры и аккордеон ЗУН.
 *
 * Перенесено из TeacherDashboard.tsx без изменений разметки и классов.
 * Состояние форм остаётся в контейнере и приходит пропсами.
 */
import type { CompFilter } from "../teacherFilters";
import { plural } from "./helpers";
import type { CompErr, Competency, DisciplineDetail, KsaEdit, ZunForm } from "./types";

interface ZunSectionProps {
  selected: DisciplineDetail;
  compForm: boolean;
  setCompForm: (value: boolean) => void;
  compErr: CompErr;
  setCompErr: (value: CompErr) => void;
  compCode: string;
  setCompCode: (value: string) => void;
  compName: string;
  setCompName: (value: string) => void;
  compSaving: boolean;
  saveCompetency: () => Promise<void>;
  compSearch: string;
  setCompSearch: (value: string) => void;
  compFilter: CompFilter;
  setCompFilter: (value: CompFilter) => void;
  compCounts: Record<CompFilter, number> | null;
  compsVisible: Competency[];
  compFiltersActive: boolean;
  openComp: string | null;
  setOpenComp: (value: string | null) => void;
  zunForm: ZunForm | null;
  setZunForm: (value: ZunForm | null) => void;
  zunMsg: string;
  setZunMsg: (value: string) => void;
  zunSaving: boolean;
  addZunEntry: () => Promise<void>;
  ksaEditing: KsaEdit | null;
  setKsaEditing: (value: KsaEdit | null) => void;
  saveKsaEdit: () => Promise<void>;
  delKsa: (id: string) => Promise<void>;
}

export function ZunSection({
  selected,
  compForm,
  setCompForm,
  compErr,
  setCompErr,
  compCode,
  setCompCode,
  compName,
  setCompName,
  compSaving,
  saveCompetency,
  compSearch,
  setCompSearch,
  compFilter,
  setCompFilter,
  compCounts,
  compsVisible,
  compFiltersActive,
  openComp,
  setOpenComp,
  zunForm,
  setZunForm,
  zunMsg,
  setZunMsg,
  zunSaving,
  addZunEntry,
  ksaEditing,
  setKsaEditing,
  saveKsaEdit,
  delKsa,
}: ZunSectionProps) {
  return (
    <>
      <div className="mb-3">
        {!compForm ? (
          <button onClick={() => { setCompForm(true); setCompErr({}); }}
            className="text-xs text-success-foreground bg-success hover:brightness-110 transition-colors px-3 py-1.5 rounded-lg cursor-pointer border-0">
            + Компетенция
          </button>
        ) : (
          <div className="rounded-lg border border-success/30 bg-success-soft p-3 space-y-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <input value={compCode} onChange={(e) => setCompCode(e.target.value)}
                  placeholder="Код: ПК-2.1"
                  aria-label="Код компетенции"
                  className={`h-9 w-full px-2 rounded border bg-input-background text-xs ${compErr.code ? "border-destructive" : "border-border-strong"}`} />
                {compErr.code && <div className="text-xs text-destructive mt-1">{compErr.code}</div>}
              </div>
              <div>
                <input value={compName} onChange={(e) => setCompName(e.target.value)}
                  placeholder="Название (необязательно)"
                  aria-label="Название компетенции"
                  className={`h-9 w-full px-2 rounded border bg-input-background text-xs ${compErr.name ? "border-destructive" : "border-border-strong"}`} />
                {compErr.name && <div className="text-xs text-destructive mt-1">{compErr.name}</div>}
              </div>
            </div>
            {compErr.msg && <div className="text-xs text-destructive">{compErr.msg}</div>}
            <div className="flex items-center gap-2">
              <button onClick={saveCompetency} disabled={compSaving}
                className="bg-success text-success-foreground px-3 py-1.5 rounded-lg hover:brightness-110 transition-colors cursor-pointer border-0 disabled:opacity-50">
                {compSaving ? "..." : "Создать"}
              </button>
              <button onClick={() => setCompForm(false)}
                className="text-xs text-muted-foreground hover:text-foreground border-0 bg-transparent cursor-pointer">Отмена</button>
            </div>
            <div className="text-xs text-subtle-foreground">ЗУН и инструменты дописываются после создания: +ЗУН у компетенции, инструменты - через Администрирование → Навыки.</div>
          </div>
        )}
      </div>

      {/* Фильтры по компетенциям: длинные списки (10+) без поиска
          приходится прокручивать вслепую. */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          placeholder="Поиск: код или название…"
          aria-label="Поиск компетенции"
          value={compSearch}
          onChange={(e) => setCompSearch(e.target.value)}
          className="h-8 w-56 px-2.5 rounded-md text-xs box-border bg-card border border-border text-foreground outline-none"
        />
        {([
          ["all", "Все", "Без дополнительного фильтра"],
          ["no-ksa", "Без ЗУН", "Компетенции без единого пункта знаний/умений/навыков — часто требуют дозаполнения"],
          ["with-ksa", "С ЗУН", "Компетенции хотя бы с одним пунктом знаний/умений/навыков"],
        ] as const).map(([id, label, hint]) => (
          <button
            key={id}
            onClick={() => setCompFilter(id)}
            aria-pressed={compFilter === id}
            title={hint}
            className={
              "text-xs px-2 py-0.5 rounded border cursor-pointer " +
              (compFilter === id
                ? "border-primary/60 bg-primary-soft text-primary-soft-foreground"
                : "border-border-strong bg-card text-muted-foreground")
            }
          >
            {label}{compCounts ? ` (${compCounts[id]})` : ""}
          </button>
        ))}
        <span className="text-[11px] text-subtle-foreground ml-auto whitespace-nowrap">
          {compsVisible.length} из {selected.competencies.length}
        </span>
        {compFiltersActive && (
          <button
            onClick={() => { setCompSearch(""); setCompFilter("all"); }}
            className="text-[11px] text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer underline whitespace-nowrap"
          >
            Сбросить
          </button>
        )}
      </div>
      {compsVisible.length === 0 && (
        <div className="text-center text-xs text-subtle-foreground py-6 bg-muted rounded-lg">
          По фильтру ничего не найдено —{" "}
          <button
            onClick={() => { setCompSearch(""); setCompFilter("all"); }}
            className="text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer underline"
          >
            сбросить
          </button>
        </div>
      )}

      {compsVisible.map((comp) => {
        const ksa = comp.ksa;
        const hasGroups = !!ksa && (
          (ksa.knowledge?.length || 0) + (ksa.abilities?.length || 0) + (ksa.skills?.length || 0)
        ) > 0;
        const total = hasGroups
          ? (ksa!.knowledge.length + ksa!.abilities.length + ksa!.skills.length)
          : comp.skills.length;
        const groups = hasGroups ? [
          { title: "Знания", items: ksa!.knowledge },
          { title: "Умения", items: ksa!.abilities },
          { title: "Навыки", items: ksa!.skills },
        ] : [];
        // Аккордеон: развёрнута максимум одна компетенция, по умолчанию
        // все свёрнуты — список из 10+ развёрнутых карточек и перегружал экран.
        const compOpen = openComp === comp.id;
        return (
        <div
          key={comp.code}
          className="mb-2 border border-border rounded-lg overflow-hidden"
        >
          <div
            role="button"
            tabIndex={0}
            aria-expanded={compOpen}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpenComp(compOpen ? null : comp.id); } }}
            onClick={() => setOpenComp(compOpen ? null : comp.id)}
            className="px-4 py-2.5 bg-muted flex items-center gap-2 cursor-pointer"
          >
            <span aria-hidden="true" className="text-subtle-foreground text-xs shrink-0">{compOpen ? "▾" : "▸"}</span>
            <span className="font-semibold text-sm text-primary shrink-0">
              {comp.code}
            </span>
            {comp.name && (
              <span className="text-xs text-muted-foreground truncate" title={comp.name}>
                {comp.name}
              </span>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setZunForm({ compId: comp.id, ksaType: "skills", text: "" });
                setZunMsg("");
                setOpenComp(comp.id);
              }}
              className="ml-auto text-xs text-primary hover:text-primary/80 border-0 bg-transparent cursor-pointer shrink-0"
              title="Добавить пункт (знание / умение / навык) в эту компетенцию"
            >
              + ЗУН
            </button>
            <span className="text-xs text-subtle-foreground shrink-0">
              {total} {plural(total, "пункт", "пункта", "пунктов")}
            </span>
          </div>
          {compOpen && (
          <div className="px-4 py-2">
          {zunForm && zunForm.compId === comp.id && (
            <div className="mb-2 rounded-lg border border-primary/30 bg-primary-soft p-2">
              <div className="flex gap-2 mb-2">
                <select
                  value={zunForm.ksaType}
                  onChange={(e) => setZunForm({ ...zunForm, ksaType: e.target.value })}
                  className="text-xs border border-border-strong rounded-md px-2 py-1 bg-input-background"
                >
                  <option value="knowledge">Знания</option>
                  <option value="abilities">Умения</option>
                  <option value="skills">Навыки</option>
                </select>
                <button
                  onClick={() => setZunForm(null)}
                  className="text-xs text-muted-foreground hover:text-foreground border-0 bg-transparent cursor-pointer"
                >
                  Отмена
                </button>
              </div>
              <textarea
                value={zunForm.text}
                onChange={(e) => setZunForm({ ...zunForm, text: e.target.value })}
                rows={3}
                maxLength={2000}
                placeholder="Текст пункта: знание, умение или навык…"
                className="w-full text-xs border border-border-strong rounded-md px-2 py-1 mb-2"
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={addZunEntry}
                  disabled={zunSaving || !zunForm.text.trim()}
                  className="text-xs bg-primary text-primary-foreground px-3 py-1.5 rounded-lg hover:bg-primary transition-colors cursor-pointer border-0 disabled:opacity-50"
                >
                  {zunSaving ? "Сохранение…" : "Добавить"}
                </button>
                {zunMsg && <span className="text-xs text-muted-foreground">{zunMsg}</span>}
              </div>
              <div className="text-xs text-subtle-foreground mt-1">Пункт попадёт в анализ при следующем пересчёте.</div>
            </div>
          )}
            {total === 0 ? (
              <div className="text-xs text-subtle-foreground">Навыки не извлечены</div>
            ) : hasGroups ? (
              groups.map((g) => g.items.length > 0 && (
                <div key={g.title} className="mb-2 last:mb-0">
                  <div className="text-xs font-semibold text-primary uppercase tracking-wide mt-1.5 mb-1">
                    {g.title} ({g.items.length})
                  </div>
                  {g.items.map((sk) => (
                    <div key={sk.id} className="py-0.5 text-xs leading-relaxed border-b border-border last:border-0">
                      {ksaEditing && ksaEditing.id === sk.id ? (
                        <div className="flex gap-2 items-start">
                          <textarea
                            value={ksaEditing.text}
                            onChange={(e) => setKsaEditing({ ...ksaEditing, text: e.target.value })}
                            rows={2}
                            maxLength={2000}
                            className="flex-1 text-xs border border-border-strong rounded-md px-2 py-1"
                          />
                          <button onClick={saveKsaEdit} disabled={zunSaving} className="text-xs bg-primary text-primary-foreground px-2 py-1 rounded-md hover:bg-primary cursor-pointer border-0 disabled:opacity-50">OK</button>
                          <button onClick={() => setKsaEditing(null)} className="text-xs text-muted-foreground hover:text-foreground border-0 bg-transparent cursor-pointer">Отмена</button>
                        </div>
                      ) : (
                        <div className="flex gap-1 items-start group">
                          <span className="flex-1">{sk.text}</span>
                          <button onClick={() => { setKsaEditing({ id: sk.id, text: sk.text }); setZunMsg(""); }} title="Редактировать" className="text-subtle-foreground hover:text-primary border-0 bg-transparent cursor-pointer text-xs">Изменить</button>
                          <button onClick={() => delKsa(sk.id)} title="Удалить" className="text-subtle-foreground hover:text-destructive border-0 bg-transparent cursor-pointer text-xs">Удалить</button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))
            ) : (
              comp.skills.map((sk, i) => (
                <div key={i} className="py-0.5 text-xs leading-relaxed border-b border-border last:border-0">
                  {sk}
                </div>
              ))
            )}
          </div>
          )}
        </div>
        );
      })}
    </>
  );
}
