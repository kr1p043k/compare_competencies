import { motion } from "motion/react";
import { Heart, Code, Sparkles } from "lucide-react";

export function Footer() {
  return (
    <motion.footer
      className="relative mt-20 py-12 border-t border-slate-200/50 dark:border-slate-700/50 bg-white/30 dark:bg-slate-900/30 backdrop-blur-xl"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.5 }}
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Sparkles className="size-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-white">
                Gap Analyzer
              </h3>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              Платформа для анализа соответствия образовательных программ
              требованиям рынка труда с использованием машинного обучения.
            </p>
          </div>

          {/* Features */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-slate-800 dark:text-white uppercase tracking-wider">
              Возможности
            </h4>
            <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-blue-600 rounded-full" />
                Сбор данных с hh.ru
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-purple-600 rounded-full" />
                ML-кластеризация вакансий
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-pink-600 rounded-full" />
                Gap-анализ компетенций
              </li>
              <li className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-600 rounded-full" />
                Интерактивная визуализация
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-slate-800 dark:text-white uppercase tracking-wider">
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
                <span
                  key={tech}
                  className="px-3 py-1 text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-full border border-slate-200 dark:border-slate-700"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-8 border-t border-slate-200/50 dark:border-slate-700/50">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
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
                <Heart className="size-4 text-red-500 fill-red-500" />
              </motion.div>
              <span>командой</span>
              <Code className="size-4" />
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-500">
              © 2026 Competency Gap Analyzer. Все права защищены.
            </p>
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
