import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Newspaper } from "lucide-react";

export function ArticlesPage() {
  return (
    <Card className="border border-gray-200 shadow-sm">
      <CardHeader className="border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-lg">
            <Newspaper className="size-5 text-white" />
          </div>
          <div>
            <CardTitle className="text-xl font-semibold text-gray-900">
              Статьи
            </CardTitle>
            <CardDescription className="text-sm text-gray-600">
              Полезные материалы и рекомендации
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        <div className="flex flex-col items-center justify-center py-16 text-gray-400">
          <Newspaper className="size-12 mb-4" />
          <p className="text-lg font-medium">Раздел готовится</p>
          <p className="text-sm mt-1">Здесь будут опубликованы статьи и аналитические материалы</p>
        </div>
      </CardContent>
    </Card>
  );
}
