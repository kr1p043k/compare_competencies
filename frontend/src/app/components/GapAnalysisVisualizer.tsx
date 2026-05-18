import { useState } from "react";
import { motion } from "motion/react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Badge } from "./ui/badge";
import {
  BarChart3,
  Activity,
  Radar,
  Flame,
  TrendingDown,
  Network,
  Download,
  Maximize2,
  Loader2,
  AlertCircle,
} from "lucide-react";

interface GapAnalysisVisualizerProps {
  profile: string;
}

type ImageType = "radar" | "ml_importance" | "cluster_insights" | "deficits";

interface ImageData {
  type: ImageType;
  title: string;
  description: string;
  icon: any;
  gradient: string;
}

const IMAGE_CONFIGS: ImageData[] = [
  {
    type: "radar",
    title: "Радарная диаграмма",
    description: "Сравнение компетенций по категориям",
    icon: Radar,
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    type: "ml_importance",
    title: "Важность признаков ML",
    description: "Приоритизация компетенций по модели",
    icon: Activity,
    gradient: "from-purple-500 to-pink-500",
  },
  {
    type: "cluster_insights",
    title: "Кластерные инсайты",
    description: "Группировка схожих навыков",
    icon: Network,
    gradient: "from-emerald-500 to-teal-500",
  },
  {
    type: "deficits",
    title: "Дефициты компетенций",
    description: "Анализ недостающих навыков",
    icon: TrendingDown,
    gradient: "from-orange-500 to-red-500",
  },
];

export function GapAnalysisVisualizer({ profile }: GapAnalysisVisualizerProps) {
  const [selectedImage, setSelectedImage] = useState<ImageType>("radar");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const imageUrl = `/api/results/images/${profile}/${selectedImage}`;
  const coverageUrl = "/api/results/images/coverage-comparison";
  const heatmapUrl = "/api/results/images/skills-heatmap";
  const correlationUrl = "/api/results/images/skill-correlation";

  const handleImageLoad = () => {
    setImageLoaded(true);
    setLoading(false);
    setError(null);
  };

  const handleImageError = () => {
    setImageLoaded(false);
    setLoading(false);
    setError("Изображение не найдено. Запустите GAP-анализ для генерации визуализаций.");
  };

  const handleImageChange = (type: ImageType) => {
    setSelectedImage(type);
    setLoading(true);
    setImageLoaded(false);
    setError(null);
  };

  const selectedConfig = IMAGE_CONFIGS.find((cfg) => cfg.type === selectedImage);

  return (
    <div className="space-y-6">
      {/* Main visualization card */}
      <Card className="border-0 shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden">
        <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-br from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`p-2.5 bg-gradient-to-br ${selectedConfig?.gradient} rounded-xl shadow-lg`}
              >
                {selectedConfig && <selectedConfig.icon className="size-5 text-white" />}
              </div>
              <div>
                <CardTitle className="text-2xl">Визуализация GAP-анализа</CardTitle>
                <CardDescription className="text-base mt-1">
                  {selectedConfig?.description}
                </CardDescription>
              </div>
            </div>
            <Badge
              variant="outline"
              className="text-sm font-semibold border-2 px-4 py-1.5"
            >
              {profile.toUpperCase()}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-8">
          {/* Image type selector */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
            {IMAGE_CONFIGS.map((config) => (
              <motion.div
                key={config.type}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Button
                  onClick={() => handleImageChange(config.type)}
                  variant={selectedImage === config.type ? "default" : "outline"}
                  className={`w-full h-auto flex-col items-start p-4 ${
                    selectedImage === config.type
                      ? `bg-gradient-to-br ${config.gradient} text-white border-0 shadow-lg`
                      : "border-2 hover:border-slate-400"
                  }`}
                >
                  <config.icon className="size-5 mb-2" />
                  <div className="text-xs font-semibold text-left">
                    {config.title}
                  </div>
                </Button>
              </motion.div>
            ))}
          </div>

          {/* Image display */}
          <motion.div
            className="relative bg-slate-100 dark:bg-slate-800 rounded-2xl overflow-hidden shadow-inner min-h-[500px] flex items-center justify-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-100/80 dark:bg-slate-800/80 backdrop-blur-sm z-10">
                <div className="text-center">
                  <Loader2 className="size-12 animate-spin text-blue-600 mx-auto mb-3" />
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Загрузка визуализации...
                  </p>
                </div>
              </div>
            )}

            {error && (
              <div className="absolute inset-0 flex items-center justify-center p-8">
                <div className="text-center max-w-md">
                  <div className="inline-flex p-4 bg-orange-100 dark:bg-orange-900/30 rounded-2xl mb-4">
                    <AlertCircle className="size-12 text-orange-600 dark:text-orange-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-2">
                    Визуализация недоступна
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    {error}
                  </p>
                </div>
              </div>
            )}

            {!error && (
              <img
                src={imageUrl}
                alt={selectedConfig?.title}
                onLoad={handleImageLoad}
                onError={handleImageError}
                className={`max-w-full h-auto transition-opacity duration-300 ${
                  imageLoaded ? "opacity-100" : "opacity-0"
                }`}
              />
            )}
          </motion.div>

          {/* Action buttons */}
          {imageLoaded && (
            <motion.div
              className="flex gap-3 mt-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <Button
                variant="outline"
                size="sm"
                className="border-2"
                onClick={() => window.open(imageUrl, "_blank")}
              >
                <Maximize2 className="mr-2 size-4" />
                Открыть в полном размере
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="border-2"
                onClick={() => {
                  const link = document.createElement("a");
                  link.href = imageUrl;
                  link.download = `${selectedImage}_${profile}.png`;
                  link.click();
                }}
              >
                <Download className="mr-2 size-4" />
                Скачать
              </Button>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Global comparisons */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <GlobalVisualizationCard
          title="Сравнение покрытия"
          description="Сопоставление всех профилей"
          imageUrl={coverageUrl}
          icon={BarChart3}
          gradient="from-blue-500 to-cyan-500"
        />
        <GlobalVisualizationCard
          title="Тепловая карта навыков"
          description="Распределение компетенций"
          imageUrl={heatmapUrl}
          icon={Flame}
          gradient="from-orange-500 to-red-500"
        />
        <GlobalVisualizationCard
          title="Корреляция навыков"
          description="Взаимосвязь компетенций"
          imageUrl={correlationUrl}
          icon={Network}
          gradient="from-purple-500 to-pink-500"
        />
      </div>
    </div>
  );
}

interface GlobalVisualizationCardProps {
  title: string;
  description: string;
  imageUrl: string;
  icon: any;
  gradient: string;
}

function GlobalVisualizationCard({
  title,
  description,
  imageUrl,
  icon: Icon,
  gradient,
}: GlobalVisualizationCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [error, setError] = useState(false);

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden h-full">
        <CardHeader className="pb-4 bg-gradient-to-br from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <div className="flex items-center gap-3">
            <div className={`p-2 bg-gradient-to-br ${gradient} rounded-lg shadow-md`}>
              <Icon className="size-4 text-white" />
            </div>
            <div>
              <CardTitle className="text-base">{title}</CardTitle>
              <CardDescription className="text-xs mt-0.5">
                {description}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-4">
          <div className="relative bg-slate-100 dark:bg-slate-800 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
            {!imageLoaded && !error && (
              <Loader2 className="size-8 animate-spin text-blue-600" />
            )}
            {error && (
              <div className="text-center p-4">
                <AlertCircle className="size-8 text-orange-600 dark:text-orange-400 mx-auto mb-2" />
                <p className="text-xs text-slate-600 dark:text-slate-400">
                  Недоступно
                </p>
              </div>
            )}
            <img
              src={imageUrl}
              alt={title}
              onLoad={() => setImageLoaded(true)}
              onError={() => setError(true)}
              className={`max-w-full h-auto transition-opacity duration-300 ${
                imageLoaded ? "opacity-100" : "opacity-0"
              }`}
            />
          </div>
          {imageLoaded && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full mt-3 text-xs hover:bg-slate-100 dark:hover:bg-slate-800"
              onClick={() => window.open(imageUrl, "_blank")}
            >
              <Maximize2 className="mr-2 size-3" />
              Открыть
            </Button>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
