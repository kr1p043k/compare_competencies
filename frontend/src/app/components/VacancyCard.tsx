import { motion } from "motion/react";
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
  TrendingUp,
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
  snippet?: {
    requirement?: string;
    responsibility?: string;
  };
}

interface VacancyCardProps {
  vacancy: Vacancy;
  onViewDetails?: (id: string) => void;
}

const experienceLevels = {
  junior: { label: "Junior", color: "from-blue-500 to-cyan-500", badge: "secondary" },
  middle: { label: "Middle", color: "from-purple-500 to-pink-500", badge: "default" },
  senior: { label: "Senior", color: "from-orange-500 to-red-500", badge: "destructive" },
};

export function VacancyCard({ vacancy, onViewDetails }: VacancyCardProps) {
  const expLevel = experienceLevels[vacancy.experience as keyof typeof experienceLevels] || experienceLevels.middle;

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
                  <span dangerouslySetInnerHTML={{ __html: vacancy.snippet.requirement }} />
                </div>
              )}
              {vacancy.snippet.responsibility && (
                <div className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">Обязанности:</span>{" "}
                  <span dangerouslySetInnerHTML={{ __html: vacancy.snippet.responsibility }} />
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
                onClick={() => onViewDetails?.(vacancy.id)}
              >
                <TrendingUp className="mr-2 size-4 group-hover/btn:rotate-12 transition-transform" />
                Подробнее
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
