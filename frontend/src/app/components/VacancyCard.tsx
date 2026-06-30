import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "./ui/card";
import {
  Building2,
  MapPin,
  Calendar,
  ExternalLink,
  Briefcase,
  DollarSign,
  Star,
  AlertTriangle,
  ChevronDown,
  Loader2,
  FileText,
  Tags,
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
}

const experienceLevels = {
  junior: { label: "Junior", color: "from-blue-500 to-cyan-500", badge: "secondary" },
  middle: { label: "Middle", color: "from-purple-500 to-pink-500", badge: "default" },
  senior: { label: "Senior", color: "from-orange-500 to-red-500", badge: "destructive" },
};

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
  "1С","1С:Предприятие","1С:Розница","1С:Бухгалтерия","1С:ЗУП","БСП","СКД",
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

function sanitizeHtml(html: string): string {
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

function parseSkillsFromHtml(html: string): string[] {
  const text = html.replace(/<[^>]+>/g, " ").replace(/&[^;]+;/g, " ").replace(/\s+/g, " ").trim();
  const found = new Set<string>();

  for (const kw of TECH_KEYWORDS) {
    const re = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    if (re.test(text) && isValidSkill(kw)) found.add(kw);
  }

  const parts = text.split(/[,;•\n\r]+/).map(s => s.trim()).filter(Boolean);
  for (const part of parts) {
    const clean = part.replace(/^[\s\-—–•*.:;]+/, "").replace(/[\s\-—–•*.:;]+$/, "");
    if (!isValidSkill(clean)) continue;
    const ws = clean.split(/\s+/);
    const capWords = ws.filter(w => /^[A-ZА-Я]/.test(w));
    if (capWords.length > 0 && capWords.length === ws.length) {
      found.add(clean);
    }
  }

  return Array.from(found).sort();
}

interface VacancyDetail {
  id: string;
  name?: string;
  description?: string;
  experience?: any;
  salary?: any;
  employer?: any;
  area?: any;
  published_at?: string;
  alternate_url?: string;
  skills?: string[];
  schedule?: any;
  employment?: any;
  key_skills?: any[];
  snippet?: any;
}

export function VacancyCard({ vacancy }: VacancyCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<VacancyDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const expLevel = experienceLevels[vacancy.experience as keyof typeof experienceLevels] || experienceLevels.middle;

  useEffect(() => {
    if (!expanded) return;
    if (detail) return;
    setLoadingDetail(true);
    fetch(`/api/vacancies/${vacancy.id}`)
      .then((r) => r.json())
      .then((d) => setDetail(d))
      .catch(() => {})
      .finally(() => setLoadingDetail(false));
  }, [expanded]);

  const formatSalary = () => {
    if (!vacancy.salary_from && !vacancy.salary_to) return null;

    const format = (num: number) => {
      return new Intl.NumberFormat("ru-RU").format(num);
    };

    const currency = vacancy.salary_currency === "RUR" ? "₽" : vacancy.salary_currency;

    if (vacancy.salary_from && vacancy.salary_to) {
      return `${format(vacancy.salary_from)} - ${format(vacancy.salary_to)} ${currency}`;
    } else if (vacancy.salary_from) {
      return `от ${format(vacancy.salary_from)} ${currency}`;
    } else {
      return `до ${format(vacancy.salary_to!)} ${currency}`;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) return "Сегодня";
    if (diffDays === 2) return "Вчера";
    if (diffDays <= 7) return `${diffDays} дня назад`;
    return date.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  };

  const salary = formatSalary();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="group"
    >
      <Card className="border-0 shadow-lg hover:shadow-2xl transition-all duration-300 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden relative">
        {/* Accent bar */}
        <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${expLevel.color}`} />

        {/* Hover glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-purple-500/0 to-pink-500/0 group-hover:from-blue-500/5 group-hover:via-purple-500/5 group-hover:to-pink-500/5 transition-all duration-500 pointer-events-none" />

        <CardHeader className="pb-4 relative">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <motion.h3
                className="text-xl font-bold text-slate-900 dark:text-white mb-2 line-clamp-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors cursor-pointer"
                whileHover={{ x: 2 }}
              >
                {vacancy.name}
              </motion.h3>
              <div className="flex items-center gap-3 flex-wrap">
                <Badge
                  variant={expLevel.badge as any}
                  className={`bg-gradient-to-r ${expLevel.color} text-white border-0 shadow-md`}
                >
                  <Briefcase className="size-3 mr-1" />
                  {expLevel.label}
                </Badge>
                {vacancy.is_spam && (
                  <Badge
                    variant="destructive"
                    className="bg-red-500/90 text-white border-0 shadow-md"
                    title={vacancy.spam_reason || ""}
                  >
                    <AlertTriangle className="size-3 mr-1" />
                    Спам
                  </Badge>
                )}
                {salary && (
                  <motion.div
                    className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold text-lg"
                    whileHover={{ scale: 1.05 }}
                  >
                    <DollarSign className="size-5" />
                    {salary}
                  </motion.div>
                )}
              </div>
            </div>

            {/* Company logo */}
            {vacancy.employer_logo ? (
              <motion.div
                className="size-16 rounded-xl overflow-hidden bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 shadow-md flex-shrink-0"
                whileHover={{ scale: 1.05, rotate: 2 }}
              >
                <img
                  src={vacancy.employer_logo}
                  alt={vacancy.employer_name}
                  className="size-full object-contain p-2"
                />
              </motion.div>
            ) : (
              <motion.div
                className={`size-16 rounded-xl bg-gradient-to-br ${expLevel.color} flex items-center justify-center shadow-lg flex-shrink-0`}
                whileHover={{ scale: 1.05, rotate: -2 }}
              >
                <Building2 className="size-8 text-white" />
              </motion.div>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4 relative">
          {/* Company and location */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <div className="p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <Building2 className="size-4 text-blue-600 dark:text-blue-400" />
              </div>
              <span className="font-medium">{vacancy.employer_name}</span>
            </div>
            <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 text-sm">
              <div className="p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                <MapPin className="size-4 text-purple-600 dark:text-purple-400" />
              </div>
              <span>{vacancy.area}</span>
              <span className="text-slate-400">•</span>
              <Calendar className="size-4" />
              <span>{formatDate(vacancy.published_at)}</span>
            </div>
          </div>

          {/* Snippet */}
          {vacancy.snippet && (vacancy.snippet.requirement || vacancy.snippet.responsibility) && (
            <div className="space-y-2">
              {vacancy.snippet.requirement && (
                <div className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">Требования:</span>{" "}
                  <span>{sanitizeHtml(vacancy.snippet.requirement)}</span>
                </div>
              )}
              {vacancy.snippet.responsibility && (
                <div className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">Обязанности:</span>{" "}
                  <span>{sanitizeHtml(vacancy.snippet.responsibility)}</span>
                </div>
              )}
            </div>
          )}

          {/* Skills */}
          {vacancy.skills && vacancy.skills.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider">
                <Star className="size-3" />
                Ключевые навыки
              </div>
              <div className="flex flex-wrap gap-2">
                {vacancy.skills.slice(0, 8).map((skill, index) => (
                  <motion.div
                    key={skill}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <Badge
                      variant="outline"
                      className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-700 border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-md transition-all"
                    >
                      {skill}
                    </Badge>
                  </motion.div>
                ))}
                {vacancy.skills.length > 8 && (
                  <Badge variant="secondary" className="bg-slate-200 dark:bg-slate-700">
                    +{vacancy.skills.length - 8}
                  </Badge>
                )}
              </div>
            </div>
          )}
        </CardContent>

        {/* Expanded details */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden border-t border-slate-200/50 dark:border-slate-700/50"
            >
              <div className="p-4 space-y-4">
                {loadingDetail ? (
                  <div className="flex items-center justify-center py-6">
                    <Loader2 className="size-5 animate-spin text-slate-400" />
                  </div>
                ) : (
                  <>
                    {/* Description */}
                    {detail?.description && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                          <FileText className="size-3" />
                          Описание вакансии
                        </div>
                        <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                          <div className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-line">
                            {sanitizeHtml(detail.description)}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* HH key_skills */}
                    {detail?.key_skills && detail.key_skills.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                          <Star className="size-3" />
                          Ключевые навыки (HH)
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {(detail.key_skills as any[]).map((ks: any) => (
                            <Badge
                              key={typeof ks === 'string' ? ks : ks.name}
                              variant="secondary"
                              className="bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800"
                            >
                              {typeof ks === 'string' ? ks : ks.name}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Skills from description (parsed fallback) */}
                    {(() => {
                      const extracted = detail?.skills ?? [];
                      const parsed = detail?.description ? parseSkillsFromHtml(detail.description) : [];
                      const displaySkills = extracted.length > 0 ? extracted : parsed;
                      if (displaySkills.length === 0) return null;
                      return (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                            <Tags className="size-3" />
                            {extracted.length > 0 ? "Найденные навыки" : "Технологии из описания"}
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {displaySkills.map((skill: string) => (
                              <Badge
                                key={skill}
                                variant="outline"
                                className="bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800"
                              >
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      );
                    })()}
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <CardFooter className="pt-4 border-t border-slate-200/50 dark:border-slate-700/50 relative">
          <div className="flex gap-2 w-full">
            <motion.div
              className="flex-1"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                variant="outline"
                className="w-full border-2 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950/30 transition-all group/btn"
                onClick={() => setExpanded((p) => !p)}
              >
                <ChevronDown className={`mr-2 size-4 transition-transform ${expanded ? "rotate-180" : ""}`} />
                {expanded ? "Свернуть" : "Подробнее"}
              </Button>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                asChild
                className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 text-white shadow-lg hover:shadow-xl transition-all"
              >
                <a href={vacancy.alternate_url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="size-4" />
                </a>
              </Button>
            </motion.div>
          </div>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
