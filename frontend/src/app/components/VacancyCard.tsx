import {
  AlertTriangle,
  Building2,
  Calendar,
  ChevronRight,
  ExternalLink,
  MapPin,
} from "lucide-react";

import { Badge, Button, Card } from "../../design";

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

// Уровень - это смысловой бейдж, а не украшение: тон Badge из дизайн-системы
// кодирует грейд (info/primary/warning), поэтому Junior/Middle/Senior нельзя
// рисовать одним нейтральным. Тона заданы токенами и работают в обеих темах.
const experienceLevels = {
  junior: { label: "Junior", tone: "info" as const },
  middle: { label: "Middle", tone: "primary" as const },
  senior: { label: "Senior", tone: "warning" as const },
  unknown: { label: "Уровень не указан", tone: "neutral" as const },
};

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

// \b в JS считает кириллицу не-буквой, поэтому в «1С» срабатывала
// односимвольная стоп-единица «с» и навык отбрасывался. Юникод-границы.
const RUSSIAN_STOPWORDS = new RegExp(
  "(?<![\\p{L}\\p{N}_])(?:"
    + "и|в|на|по|с|для|от|за|из|у|о|об|про|без|до|при|не|или|а|но|да|же|ли|бы|если|чтобы|так|как|это|что|котор|таких|такой|такие|всех|все|всё|может|можно|навыки|опыт|знание|понимание|умение|работа|разработка|настройка|внедрение|поддержка|сопровождение|управление|взаимодействие|наличие|готовность|способность|участие|проведение|создание|использование|обеспечение|выполнение|формирование|организация|обучение|контроль|оценка|анализ|расчет|подготовка|применение|интеграция|автоматизация|оптимизация|проектирование|администрирование|конфигурирование|программирование|тестирование|отладка|документирование|коммуникабельность|системное|аналитическое|критическое|техническое|проактивность|ответственность|самостоятельность|ориентированность|стрессоустойчивость|исполнительность|дисциплинированность|пунктуальность|работоспособность|обучаемость|грамотность|аккуратность|внимательность|терпеливость|честность|порядочность|креативность|инициативность|целеустремленность|нацеленность|мотивация|интерес|желание|готов|уверенный|уверенное|хорошее|базовое|высшее|среднее|полное|неполное|специальное|профессиональное|образование|зарплата|доход|график|офис|удаленно|гибрид|командировки|оформление|тк|рф|сетью|точками|узлов|области|данными|системами|средой|платформой|архитектурой|пользователями|задачами|проектами|командами|процессами|требованиями|решениями|результатами|целями|сроками|стандартами|регламентами|инструментами|технологиями|методами|подходами|принципами|механизмами|алгоритмами|протоколами|форматами|типами"
    + ")(?![\\p{L}\\p{N}_])",
  "iu",
);

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
    <div className="grid size-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-border bg-surface">
      {vacancy.employer_logo ? (
        <img
          src={vacancy.employer_logo}
          alt={vacancy.employer_name}
          className="size-full object-contain p-1.5"
          loading="lazy"
        />
      ) : (
        <Building2 className="size-5 text-muted-foreground" />
      )}
    </div>
  );

  const LevelChip = (
    <Badge tone={expLevel.tone} title={`Уровень: ${expLevel.label}`}>
      <span
        className="size-1.5 shrink-0 rounded-full bg-current opacity-70"
        aria-hidden="true"
      />
      {expLevel.label}
    </Badge>
  );

  const SpamChip = vacancy.is_spam ? (
    <Badge tone="danger" title={vacancy.spam_reason || ""}>
      <AlertTriangle className="size-3" />
      Спам
    </Badge>
  ) : null;

  const SkillChips = skills.length > 0 && (
    <div className="flex flex-wrap gap-1">
      {skills.slice(0, 7).map((skill) => (
        <Badge key={skill} tone="neutral">
          {skill}
        </Badge>
      ))}
      {skills.length > 7 && (
        <span className="px-1.5 py-0.5 text-xs text-subtle-foreground">
          +{skills.length - 7}
        </span>
      )}
    </div>
  );

  const Actions = (
    <div className="flex items-center gap-2">
      <Button
        variant="secondary"
        size="sm"
        className="flex-1 justify-center"
        onClick={() => onOpen?.(vacancy)}
      >
        Подробнее
        <ChevronRight className="ml-1 size-3.5" />
      </Button>
      <Button variant="ghost" size="sm" className="size-8 shrink-0 p-0" asChild>
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
      <Card
        tone="flat"
        className="flex flex-row items-start gap-4 gap-y-0 p-4 transition-colors hover:border-border-strong"
      >
        {Logo}
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-[15px] font-semibold text-foreground">
              {vacancy.name}
            </h3>
            {LevelChip}
            {SpamChip}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
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
          <span className="whitespace-nowrap text-sm font-semibold text-foreground">
            {salary ?? <span className="font-normal text-subtle-foreground">не указана</span>}
          </span>
          <div className="w-[150px]">{Actions}</div>
        </div>
      </Card>
    );
  }

  // ---------------------------------------------------------------- grid
  return (
    <Card
      tone="flat"
      className="flex h-full flex-col gap-0 p-4 transition-colors hover:border-border-strong"
    >
      <div className="flex items-start gap-3">
        {Logo}
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 text-[15px] font-semibold leading-snug text-foreground">
            {vacancy.name}
          </h3>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
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
        <span className="ml-auto text-sm font-semibold text-foreground">
          {salary ?? <span className="font-normal text-subtle-foreground">з/п не указана</span>}
        </span>
      </div>

      {snippet && (
        <p className="mt-3 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {sanitizeHtml(snippet)}
        </p>
      )}

      {skills.length > 0 && <div className="mt-3">{SkillChips}</div>}

      <div className="mt-auto pt-3">
        <div className="flex items-center justify-between gap-2 text-xs text-subtle-foreground">
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
