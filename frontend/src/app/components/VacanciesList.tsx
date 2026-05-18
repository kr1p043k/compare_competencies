import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { VacancyCard } from "./VacancyCard";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  Search,
  Filter,
  Briefcase,
  TrendingUp,
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
  List,
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

interface VacanciesResponse {
  items: Vacancy[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

interface VacanciesListProps {
  onViewDetails?: (id: string) => void;
}

export function VacanciesList({ onViewDetails }: VacanciesListProps) {
  const [vacancies, setVacancies] = useState<Vacancy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [experienceFilter, setExperienceFilter] = useState<string>("all");
  const [cityFilter, setCityFilter] = useState<string>("all");
  const [availableCities, setAvailableCities] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const itemsPerPage = 12;

  useEffect(() => {
    loadVacancies();
  }, [currentPage, experienceFilter, cityFilter]);

  const loadVacancies = async () => {
    setLoading(true);
    setError(null);

    try {
      const offset = (currentPage - 1) * itemsPerPage;
      const params = new URLSearchParams({
        limit: itemsPerPage.toString(),
        offset: offset.toString(),
      });

      if (experienceFilter && experienceFilter !== "all") {
        params.append("experience", experienceFilter);
      }

      if (searchQuery.trim()) {
        params.append("search", searchQuery.trim());
      }

      const response = await fetch(`/api/vacancies?${params}`);
      if (!response.ok) {
        throw new Error("Ошибка загрузки вакансий");
      }

      const data: VacanciesResponse = await response.json();

      // Фильтрация по городу на клиенте
      let filteredItems = data.items;
      if (cityFilter && cityFilter !== "all") {
        filteredItems = data.items.filter(v => v.area === cityFilter);
      }

      setVacancies(filteredItems);
      setTotal(cityFilter !== "all" ? filteredItems.length : data.total);

      // Извлечение уникальных городов
      const cities = Array.from(new Set(data.items.map(v => v.area)))
        .filter(city => city && city !== "Не указано")
        .sort();
      setAvailableCities(cities);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    setCurrentPage(1);
    loadVacancies();
  };

  const handleSearchKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const totalPages = Math.ceil(total / itemsPerPage);

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <div className="inline-flex items-center justify-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse" />
            <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 rounded-2xl shadow-2xl">
              <Briefcase className="size-8 text-white" />
            </div>
          </div>
          <h2 className="text-4xl font-black bg-gradient-to-r from-slate-900 via-blue-800 to-purple-900 dark:from-white dark:via-blue-200 dark:to-purple-200 bg-clip-text text-transparent">
            Вакансии с hh.ru
          </h2>
        </div>
        <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Найдено <span className="font-bold text-blue-600 dark:text-blue-400">{total}</span> актуальных вакансий
        </p>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-md">
                  <Filter className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle>Фильтры и поиск</CardTitle>
                  <CardDescription>Настройте параметры для поиска вакансий</CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant={viewMode === "grid" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("grid")}
                  className="size-9"
                >
                  <LayoutGrid className="size-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("list")}
                  className="size-9"
                >
                  <List className="size-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Search */}
              <div className="md:col-span-2 space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Поиск по названию
                </label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-slate-400" />
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={handleSearchKeyPress}
                      placeholder="Введите должность или компанию..."
                      className="pl-10 h-11 border-2"
                    />
                  </div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      onClick={handleSearch}
                      disabled={loading}
                      className="h-11 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    >
                      {loading ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Search className="size-4" />
                      )}
                    </Button>
                  </motion.div>
                </div>
              </div>

              {/* Experience filter */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Уровень опыта
                </label>
                <Select value={experienceFilter} onValueChange={setExperienceFilter}>
                  <SelectTrigger className="h-11 border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Все уровни</SelectItem>
                    <SelectItem value="junior">Junior</SelectItem>
                    <SelectItem value="middle">Middle</SelectItem>
                    <SelectItem value="senior">Senior</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* City filter */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Город
                </label>
                <Select value={cityFilter} onValueChange={setCityFilter}>
                  <SelectTrigger className="h-11 border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Все города</SelectItem>
                    {availableCities.map(city => (
                      <SelectItem key={city} value={city}>
                        {city}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Active filters */}
            {(experienceFilter !== "all" || cityFilter !== "all" || searchQuery) && (
              <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-slate-200/50 dark:border-slate-700/50">
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Активные фильтры:
                </span>
                {experienceFilter !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => setExperienceFilter("all")}
                  >
                    {experienceFilter} ✕
                  </Badge>
                )}
                {cityFilter !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => setCityFilter("all")}
                  >
                    {cityFilter} ✕
                  </Badge>
                )}
                {searchQuery && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => {
                      setSearchQuery("");
                      handleSearch();
                    }}
                  >
                    "{searchQuery}" ✕
                  </Badge>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Vacancies Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center space-y-4">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Loader2 className="size-16 text-blue-600 mx-auto" />
            </motion.div>
            <p className="text-slate-600 dark:text-slate-400">Загрузка вакансий...</p>
          </div>
        </div>
      ) : error ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Card className="border-2 border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/20">
            <CardContent className="pt-6 text-center">
              <AlertCircle className="size-12 text-red-600 dark:text-red-400 mx-auto mb-3" />
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                Ошибка загрузки
              </h3>
              <p className="text-red-700 dark:text-red-300">{error}</p>
              <Button
                onClick={loadVacancies}
                variant="outline"
                className="mt-4 border-red-300 dark:border-red-700"
              >
                Попробовать снова
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      ) : vacancies.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Card className="border-2 border-slate-200 dark:border-slate-700">
            <CardContent className="pt-6 text-center py-20">
              <TrendingUp className="size-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Вакансии не найдены
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Попробуйте изменить параметры поиска
              </p>
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <>
          <motion.div
            className={`grid gap-6 ${
              viewMode === "grid"
                ? "grid-cols-1 lg:grid-cols-2"
                : "grid-cols-1"
            }`}
            layout
          >
            <AnimatePresence mode="popLayout">
              {vacancies.map((vacancy, index) => (
                <motion.div
                  key={vacancy.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <VacancyCard vacancy={vacancy} onViewDetails={onViewDetails} />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>

          {/* Pagination */}
          {totalPages > 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="flex items-center justify-center gap-2"
            >
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1 || loading}
                className="size-10"
              >
                <ChevronLeft className="size-4" />
              </Button>

              {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }

                return (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "default" : "outline"}
                    size="icon"
                    onClick={() => setCurrentPage(pageNum)}
                    disabled={loading}
                    className={`size-10 ${
                      currentPage === pageNum
                        ? "bg-gradient-to-r from-blue-600 to-purple-600"
                        : ""
                    }`}
                  >
                    {pageNum}
                  </Button>
                );
              })}

              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages || loading}
                className="size-10"
              >
                <ChevronRight className="size-4" />
              </Button>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
