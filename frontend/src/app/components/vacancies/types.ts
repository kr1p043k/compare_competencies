// Общие типы списка вакансий. Вынесены отдельным модулем, чтобы дочерние
// компоненты (карточка, фильтры, пагинация, статистика) не тянули контейнер.

export interface VacancyInfo {
  count: number;
  with_skills?: number;
  file_modified: string | null;
  date_range: { from: string; to: string } | null;
  load_error: string | null;
}

export interface Vacancy {
  id: string;
  name: string;
  experience: string;
  salary_from?: number;
  salary_to?: number;
  salary_currency?: string;
  employer_name: string;
  employer_logo?: string;
  area: string;
  published_at: string;
  alternate_url: string;
  skills: string[];
  snippet?: {
    requirement?: string;
    responsibility?: string;
  };
}

export interface VacanciesResponse {
  items: Vacancy[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
}

/** Применённые (а не черновые) фильтры, по которым уходит запрос к API. */
export interface AppliedFilters {
  search: string;
  experience: string;
  city: string;
  months: number | null;
  date_from: string;
  date_to: string;
}

export interface VacanciesListProps {
  pipelineStep?: PipelineStep | null;
  pipelineLoading?: boolean;
  restartFlag?: number;
  onStartPipeline?: (regionIds: string, profession: string, maxPages?: number, periodDays?: number) => void;
  canRunPipeline?: boolean;
}
