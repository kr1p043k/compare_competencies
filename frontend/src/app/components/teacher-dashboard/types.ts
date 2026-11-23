/**
 * Типы данных teacher-кабинета.
 *
 * Перенесено из TeacherDashboard.tsx без изменений.
 */
export type Direction = {
  dir_code: string;
  name: string;
  profile: string;
  disciplines_count: number;
};

export type DirectionAnalysis = {
  direction: string;
  direction_name: string;
  profile: string;
  total_disciplines: number;
  average_coverage: number;
  /** Охват спроса: доля релевантных рынку навыков, реально закрытых. */
  average_market_top_coverage?: number;
  /** False, когда релевантных навыков слишком мало для процента. */
  average_market_top_reliable?: boolean;
  /** Охват спроса с весом спроса по каждому навыку (качество охвата). */
  average_market_weight_coverage?: number;
  average_quality_coverage?: number;
  coverage_level: string;
  total_gaps_across_all: number;
  top_cross_discipline_gaps: { skill: string; disciplines: number }[];
  /** frequency - вакансий с навыком; change_pct/basis - только когда
   *  бэкенд нашёл динамику (SnapshotTrendAnalyzer, «рост спроса»). */
  top_emerging_across_all: { skill: string; frequency: number; change_pct?: number | null; basis?: string }[];
  recommendations: { type: string; priority: string; message: string }[];
  trends: { rising: { skill: string; change_pct: number }[]; declining: { skill: string; change_pct: number }[] };
  disciplines: { name: string; coverage_ratio: number; weighted_coverage?: number; strong_coverage?: number; coverage_level: string; gaps: number; emerging: number; emerging_growth?: number; emerging_growth_skills?: { skill: string; change_pct: number | null }[]; market_top_coverage?: number; market_weight_coverage?: number; market_top_total?: number; market_top_reliable?: boolean; market_top_source?: string; years?: number[] }[];
  generated_at: string;
};

export type Discipline = {
  id: string;
  name: string;
  in_scope?: boolean;
  scope_source?: string;
  competencies_count: number;
  skills_count: number;
  knowledge_count?: number;
  abilities_count?: number;
  semester?: number | null;
  course?: number | null;
};

export interface KsaItem {
  id: string;
  text: string;
}

export type Competency = {
  id: string;
  code: string;
  name: string;
  skills: string[];
  ksa?: {
    knowledge: KsaItem[];
    abilities: KsaItem[];
    skills: KsaItem[];
  };
};

export type DisciplineDetail = {
  id: string;
  name: string;
  competencies: Competency[];
};

export type Recommendation = {
  id: number;
  discipline_id: string;
  competency_id: string;
  suggestion: string;
  suggestion_type: string;
};

export type Stats = {
  /** Часть полей может отсутствовать, если анализ ещё не запускался. */
  total_disciplines?: number;
  total_competencies?: number;
  total_skills?: number;
  total_reports?: number;
  by_profession?: Record<string, number>;
};

export type ScopeChain = {
  total: number;
  in_scope: number;
  excluded_manual: number;
  excluded_no_data: number;
  analyzed: number | null;
};

export type RpdStatus = {
  status: string;
  error?: string;
  stats?: { progress?: number; stage?: string; disciplines?: number };
};

/** Строка анализа дисциплины из payload направления. */
export type DisciplineAnalysisRow = DirectionAnalysis["disciplines"][number];

/** Ошибки формы создания компетенции. */
export type CompErr = { code?: string; name?: string; msg?: string };

/** Форма добавления пункта ЗУН. */
export type ZunForm = { compId: string; ksaType: string; text: string };

/** Инлайн-редактирование пункта ЗУН. */
export type KsaEdit = { id: string; text: string };

/** Дисциплина в списках added/removed отчёта сравнения наборов. */
export type IntakeDisciplineCard = {
  name: string;
  coverage_ratio?: number | null;
  coverage_level?: string | null;
  gaps?: number | null;
};

/** Дисциплина, которая есть в обоих наборах, с дельтой покрытия. */
export type IntakeKeptDiscipline = {
  name: string;
  old_name?: string | null;
  coverage_ratio_old: number;
  coverage_ratio_new: number;
  coverage_delta: number;
  gaps_old: number;
  gaps_new: number;
  gaps_delta: number;
  significant_change: boolean;
};

/** Сводка отчёта сравнения наборов. */
export type IntakeSummary = {
  added: number;
  removed: number;
  kept: number;
  avg_coverage_old: number;
  avg_coverage_new: number;
  avg_coverage_delta: number;
  improved: number;
  worsened: number;
  unchanged: number;
};

/** Ответ GET /teacher/intake-comparison. */
export type IntakeComparisonReport = {
  direction: string;
  old_year: number;
  new_year: number;
  direction_name?: string | null;
  added: IntakeDisciplineCard[];
  removed: IntakeDisciplineCard[];
  kept: IntakeKeptDiscipline[];
  summary: IntakeSummary;
};
