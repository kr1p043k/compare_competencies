import { motion } from "motion/react";
import { Heart, Sparkles } from "lucide-react";

import { Badge } from "../../design";

export function Footer() {
  return (
    <motion.footer
      className="relative mt-20 py-12 border-t border-border/50 bg-surface/30"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5 }}
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="p-2 bg-primary rounded-lg">
                <Sparkles className="size-5 text-primary-foreground" aria-hidden="true" />
              </div>
              <h3 className="text-lg font-bold text-foreground">
                Gap Analyzer
              </h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Платформа для анализа соответствия образовательных программ
              требованиям рынка труда с использованием машинного обучения.
            </p>
          </div>

          {/* Features */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wider">
              Возможности
            </h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-chart-1 rounded-full" aria-hidden="true" />
                Сбор данных с hh.ru
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-chart-5 rounded-full" aria-hidden="true" />
                ML-кластеризация вакансий
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-chart-4 rounded-full" aria-hidden="true" />
                Gap-анализ компетенций
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-chart-3 rounded-full" aria-hidden="true" />
                Интерактивная визуализация
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wider">
              Технологии
            </h4>
            <div className="flex flex-wrap gap-2">
              {[
                "React",
                "TypeScript",
                "Tailwind",
                "FastAPI",
                "Python",
                "ML",
              ].map((tech) => (
                <Badge key={tech} tone="neutral" className="px-3 py-1 text-xs">
                  {tech}
                </Badge>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-8 border-t border-border/50">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>Сделано с</span>
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              >
                <Heart className="size-4 text-destructive fill-current" aria-hidden="true" />
              </motion.div>

            </div>
            <p className="text-sm text-subtle-foreground">
              © 2026 Competency Gap Analyzer. Все права защищены.
            </p>
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
