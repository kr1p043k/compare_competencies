import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { TrendingUp, Filter, GitCompare } from "lucide-react";

export function ScientificTrendsTab() {
  return (
    <div className="space-y-6">
      {/* Filters row */}
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-indigo-600 rounded-lg">
              <Filter className="size-5 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl font-semibold text-gray-900">
                Фильтры
              </CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Параметры отбора научных публикаций
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex gap-4 flex-wrap">
            <div className="h-10 w-48 bg-gray-100 rounded-md animate-pulse" />
            <div className="h-10 w-48 bg-gray-100 rounded-md animate-pulse" />
            <div className="h-10 w-32 bg-gray-100 rounded-md animate-pulse" />
            <div className="h-10 w-32 bg-gray-100 rounded-md animate-pulse" />
          </div>
        </CardContent>
      </Card>

      {/* Trends visualization placeholder */}
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-indigo-600 rounded-lg">
              <TrendingUp className="size-5 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl font-semibold text-gray-900">
                Динамика научных тем
              </CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Изменение частоты упоминаний тем в научных публикациях по годам
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-16 text-gray-400">
            <TrendingUp className="size-12 mb-4" />
            <p className="text-lg font-medium">Раздел готовится</p>
            <p className="text-sm mt-1">Здесь будет отображаться динамика научных тем по данным arXiv и OpenAlex</p>
          </div>
        </CardContent>
      </Card>

      {/* Comparison placeholder */}
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-indigo-600 rounded-lg">
              <GitCompare className="size-5 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl font-semibold text-gray-900">
                Сравнение с компетенциями ОП
              </CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Сопоставление трендов научных тем с компетенциями образовательной программы
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-16 text-gray-400">
            <GitCompare className="size-12 mb-4" />
            <p className="text-lg font-medium">Раздел готовится</p>
            <p className="text-sm mt-1">Здесь будет доступно сравнение динамики научных тем с компетенциями КРМ</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
