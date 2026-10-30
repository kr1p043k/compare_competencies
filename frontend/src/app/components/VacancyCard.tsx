import { Button } from "./ui/button";
import { Card } from "./ui/card";
import {
  Building2,
  MapPin,
  Calendar,
  ExternalLink,
  AlertTriangle,
  ChevronRight,
} from "lucide-react";

interface Vacancy {
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
  is_spam?: boolean;
  spam_reason?: string;
  snippet?: {
    requirement?: string;
    responsibility?: string;
  };
}

interface VacancyCardProps {
  vacancy: Vacancy;
  onOpen?: (vacancy: Vacancy) => void;
  /** "grid" - карточка в две колонки, "list" - компактная строка. */
  variant?: "grid" | "list";
}

// Уровень - это смысловой бейдж, а не украшение: цвет точки и подложки
// кодирует грейд, поэтому Junior/Middle/Senior нельзя рисовать одним серым.
const experienceLevels = {
  junior: {
    label: "Junior",
    dot: "bg-sky-500",
    badge:
      "bg-sky-50 text-sky-700 ring-sky-200 dark:bg-sky-500/10 dark:text-sky-300 dark:ring-sky-500/25",
  },
  middle: {
    label: "Middle",
    dot: "bg-violet-500",
    badge:
      "bg-violet-50 text-violet-700 ring-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/25",
  },
  senior: {
    label: "Senior",
    dot: "bg-amber-500",
    badge:
      "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/25",
  },
  unknown: {
    label: "Уровень не указан",
    dot: "bg-slate-400",
    badge:
      "bg-slate-50 text-slate-500 ring-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700",
  },
} as const;

const CURRENCY_SYMBOLS: Record<string, string> = {
  "RUR": "₽", "RUB": "₽", "USD": "$", "EUR": "€", "GBP": "£",
  "KZT": "₸", "BYN": "Br", "AZN": "₼", "UZS": "сум", "GEL": "₾",
  "KGS": "сом", "TJS": "смн", "UAH": "₴", "CNY": "¥",
};

function currencySymbol(code?: string): string {
  // Неизвестную валюту показываем кодом как есть, а не молча считаем рублями.
  return CURRENCY_SYMBOLS[code || ""] || code || "₽";
}

const TECH_KEYWORDS = new Set([
  "Python","PyTorch","TensorFlow","Keras","JAX","NumPy","Pandas","Scikit-learn",
  "OpenCV","Pillow","scikit-image","Docker","Kubernetes","MLFlow","ClearML",
  "WandB","Weights & Biases","YOLO","DINOv2","Qwen","VLM","ONNX","TensorRT",
  "Triton Inference Server","vLLM","Git","Jira","Confluence","SQL","NoSQL",
  "PostgreSQL","MySQL","MongoDB","Redis","Kafka","RabbitMQ","FastAPI","Flask",
  "Django","React","Vue","Angular","Node.js","TypeScript","JavaScript","HTML",
  "CSS","AWS","GCP","Azure","Linux","Bash","CI/CD","Jenkins","GitLab CI",
  "GitHub Actions","C++","Java","Go","Rust","Scala","Ruby","PHP",
  "SQLAlchemy","Alembic","Pydantic","Celery","Nginx","Gunicorn","Uvicorn",
  "Machine Learning","Deep Learning","Computer Vision","NLP","LLM","RAG",
  "Transformer","LangChain","LlamaIndex","Hugging Face","Spark","Hadoop",
  "Airflow","dbt","Kuberhealthy","Prometheus","Grafana","ELK","Elasticsearch",
  "Prolog","SAS","MATLAB","Tableau","Power BI","Excel","Word","PowerPoint",
  "Photoshop","Figma","Sketch","Illustrator","InDesign",
  "1С","1C","1С:Предприятие","1С:Розница","1С:Бухгалтерия","1С:ЗУП","1С:УТ","БСП","СКД",
  "УРИБ","РИБ","ККТ","MS-SQL","HTTP-сервисы","Веб-сервисы",
  "ЕГАИС","МДЛП","ФГИС","Честный ЗНАК","ККМ","ТСД","ЭЦП",
  "SiebelCRM","ActiveMQ","WebSocket","WebSockets","Helm","gRPC",
  "Spring Boot","Spring Cloud","Spring Security","Spring Data","Spring Framework",
  "JPA","Hibernate","WebFlux","Micrometer","JVM","JFR","JIT",
  "Circuit Breaker","Saga","Event Sourcing","CQRS","Retry","Backoff",
  "Zero-downtime","CI/CD","GitLab CI",
  "SOAP","REST","HTTP","XML","JSON","YAML","gRPC","FTP","SFTP",
  "YourKit","async-profiler",
]);

const RUSSIAN_STOPWORDS = /\b(и|в|на|по|с|для|от|за|из|у|о|об|про|без|до|при|не|или|а|но|да|же|ли|бы|если|чтобы|так|как|это|что|котор|таких|такой|такие|всех|все|всё|может|можно|навыки|опыт|знание|понимание|умение|работа|разработка|настройка|внедрение|поддержка|сопровождение|управление|взаимодействие|наличие|готовность|способность|участие|проведение|создание|использование|обеспечение|выполнение|формирование|организация|обучение|контроль|оценка|анализ|расчет|подготовка|применение|интеграция|автоматизация|оптимизация|проектирование|администрирование|конфигурирование|программирование|тестирование|отладка|документирование|коммуникабельность|системное|аналитическое|критическое|техническое|проактивность|ответственность|самостоятельность|ориентированность|стрессоустойчивость|исполнительность|дисциплинированность|пунктуальность|работоспособность|обучаемость|грамотность|аккуратность|внимательность|терпеливость|честность|порядочность|креативность|инициативность|целеустремленность|нацеленность|мотивация|интерес|желание|готов|уверенный|уверенное|хорошее|базовое|высшее|среднее|полное|неполное|специальное|профессиональное|образование|зарплата|доход|график|офис|удаленно|гибрид|командировки|оформление|тк|рф|сетью|точками|узлов|области|данными|системами|средой|платформой|архитектурой|пользователями|задачами|проектами|командами|процессами|требованиями|решениями|результатами|целями|сроками|стандартами|регламентами|инструментами|технологиями|методами|подходами|принципами|механизмами|алгоритмами|протоколами|форматами|типами)+/iu;

export function sanitizeHtml(html: string): string {
  return html
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/?[^>]+(>|$)/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function isValidSkill(s: string): boolean {
  const len = s.length;
  if (len < 2 || len > 40) return false;
  if (/^[\d\s\-_./#+]+$/.test(s)) return false;
  if (/[()[\]{}«»"':;]/.test(s)) return false;
  const words = s.split(/\s+/);
  if (words.some(w => /^[а-яё]/.test(w))) return false;
  if (words.some(w => RUSSIAN_STOPWORDS.test(w))) return false;
  return true;
}

export function parseSkillsFromHtml(html: string): string[] {
  const text = html.replace(/<[^>]+>/g, " ").replace(/&[^;]+;/g, " ").replace(/\s+/g, " ").trim();
  const found = new Set<string>();

  for (const kw of TECH_KEYWORDS) {
    // Юникод-границы: JS \b без u-флага считает кириллицу не-буквой,
    // поэтому `\b1С\b` никогда не матчится. Используем \p{L}\p{N}.
    const re = new RegExp(
      `(?<![\\p{L}\\p{N}_])${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\p{L}\\p{N}_])`,
      "iu"
    );
    if (re.test(text) && isValidSkill(kw)) found.add(kw);
  }

  const parts = text.split(/[,;•\n\r]+/).map(s => s.trim()).filter(Boolean);
  for (const part of parts) {
    const clean = part.replace(/^[\s\-*.:;]+/, "").replace(/[\s\-*.:;]+$/, "");
    if (!isValidSkill(clean)) continue;
    const ws = clean.split(/\s+/);
    const capWords = ws.filter(w => /^[A-ZА-Я]/.test(w));
    if (capWords.length > 0 && capWords.length === ws.length) {
      found.add(clean);
    }
  }

  return Array.from(found).sort();
}

export function VacancyCard({ vacancy, onOpen, variant = "grid" }: VacancyCardProps) {
  // Неизвестный уровень показываем честно, а не как "Middle": молчаливый
  // фолбэк врал в карточке (hh отдаёт между1и3 / более6 / без опыта).
  const expLevel =
    experienceLevels[vacancy.experience as keyof typeof experienceLevels] ??
    experienceLevels.unknown;

  const formatSalary = () => {
    if (!vacancy.salary_from && !vacancy.salary_to) return null;

    const format = (num: number) => new Intl.NumberFormat("ru-RU").format(num);
    const currency = currencySymbol(vacancy.salary_currency);

    if (vacancy.salary_from && vacancy.salary_to) {
      if (vacancy.salary_from === vacancy.salary_to) {
        return `${format(vacancy.salary_from)} ${currency}`;
      }
      return `${format(vacancy.salary_from)} - ${format(vacancy.salary_to)} ${currency}`;
    }
    if (vacancy.salary_from) return `от ${format(vacancy.salary_from)} ${currency}`;
    return `до ${format(vacancy.salary_to!)} ${currency}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const diffDays = Math.ceil(
      Math.abs(new Date().getTime() - date.getTime()) / (1000 * 60 * 60 * 24),
    );
    if (diffDays === 1) return "сегодня";
    if (diffDays === 2) return "вчера";
    if (diffDays <= 4) return `${diffDays} дня назад`;
    if (diffDays <= 7) return `${diffDays} дней назад`;
    return date.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  };

  const salary = formatSalary();
  const skills = (vacancy.skills || []).filter((s) => (s || "").trim());
  const snippet = vacancy.snippet?.requirement || vacancy.snippet?.responsibility;

  const Logo = (
    <div className="grid size-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
      {vacancy.employer_logo ? (
        <img
          src={vacancy.employer_logo}
          alt={vacancy.employer_name}
          className="size-full object-contain p-1.5"
          loading="lazy"
        />
      ) : (
        <Building2 className="size-5 text-slate-400 dark:text-slate-500" />
      )}
    </div>
  );

  const LevelChip = (
    <span
      title={`Уровень: ${expLevel.label}`}
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset ${expLevel.badge}`}
    >
      <span className={`size-1.5 rounded-full ${expLevel.dot}`} aria-hidden />
      {expLevel.label}
    </span>
  );

  const SpamChip = vacancy.is_spam ? (
    <span
      title={vacancy.spam_reason || ""}
      className="inline-flex items-center gap-1 rounded bg-red-50 px-1.5 py-0.5 text-[11px] font-medium text-red-700 dark:bg-red-950/40 dark:text-red-300"
    >
      <AlertTriangle className="size-3" />
      Спам
    </span>
  ) : null;

  const SkillChips = skills.length > 0 && (
    <div className="flex flex-wrap gap-1">
      {skills.slice(0, 7).map((skill) => (
        <span
          key={skill}
          className="rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
        >
          {skill}
        </span>
      ))}
      {skills.length > 7 && (
        <span className="rounded px-1.5 py-0.5 text-[11px] text-slate-400">
          +{skills.length - 7}
        </span>
      )}
    </div>
  );

  const Actions = (
    <div className="flex items-center gap-2">
      <Button
        variant="outline"
        size="sm"
        className="flex-1 justify-center"
        onClick={() => onOpen?.(vacancy)}
      >
        Подробнее
        <ChevronRight className="ml-1 size-3.5" />
      </Button>
      <Button variant="ghost" size="icon" className="size-8 shrink-0" asChild>
        <a
          href={vacancy.alternate_url}
          target="_blank"
          rel="noopener noreferrer"
          title="Открыть на hh.ru"
        >
          <ExternalLink className="size-3.5" />
        </a>
      </Button>
    </div>
  );

  // ---------------------------------------------------------------- list
  if (variant === "list") {
    return (
      <Card className="flex flex-row items-start gap-4 gap-y-0 rounded-lg border border-slate-200 bg-white p-4 shadow-none transition-colors hover:border-slate-300 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700">
        {Logo}
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-[15px] font-semibold text-slate-900 dark:text-slate-100">
              {vacancy.name}
            </h3>
            {LevelChip}
            {SpamChip}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
            <span className="truncate">{vacancy.employer_name}</span>
            <span aria-hidden>·</span>
            <span className="inline-flex items-center gap-1">
              <MapPin className="size-3" />
              {vacancy.area}
            </span>
            <span aria-hidden>·</span>
            <span className="inline-flex items-center gap-1">
              <Calendar className="size-3" />
              {formatDate(vacancy.published_at)}
            </span>
          </div>
          {SkillChips}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <span className="whitespace-nowrap text-sm font-semibold text-slate-900 dark:text-slate-100">
            {salary ?? <span className="font-normal text-slate-400">не указана</span>}
          </span>
          <div className="w-[150px]">{Actions}</div>
        </div>
      </Card>
    );
  }

  // ---------------------------------------------------------------- grid
  return (
    <Card className="flex h-full flex-col gap-0 rounded-lg border border-slate-200 bg-white p-4 shadow-none transition-colors hover:border-slate-300 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700">
      <div className="flex items-start gap-3">
        {Logo}
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 text-[15px] font-semibold leading-snug text-slate-900 dark:text-slate-100">
            {vacancy.name}
          </h3>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
            <span className="truncate">{vacancy.employer_name}</span>
            <span aria-hidden>·</span>
            <span className="inline-flex items-center gap-1">
              <MapPin className="size-3" />
              {vacancy.area}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {LevelChip}
        {SpamChip}
        <span className="ml-auto text-sm font-semibold text-slate-900 dark:text-slate-100">
          {salary ?? <span className="font-normal text-slate-400">з/п не указана</span>}
        </span>
      </div>

      {snippet && (
        <p className="mt-3 line-clamp-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          {sanitizeHtml(snippet)}
        </p>
      )}

      {skills.length > 0 && <div className="mt-3">{SkillChips}</div>}

      <div className="mt-auto pt-3">
        <div className="flex items-center justify-between gap-2 text-[11px] text-slate-400">
          <span className="inline-flex items-center gap-1">
            <Calendar className="size-3" />
            {formatDate(vacancy.published_at)}
          </span>
        </div>
        <div className="mt-2">{Actions}</div>
      </div>
    </Card>
  );
}
