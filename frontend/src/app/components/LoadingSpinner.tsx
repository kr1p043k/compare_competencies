import { motion } from "motion/react";

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center space-x-2">
      <motion.div
        className="w-2 h-2 bg-current rounded-full"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [1, 0.5, 1],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="w-2 h-2 bg-current rounded-full"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [1, 0.5, 1],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.2,
        }}
      />
      <motion.div
        className="w-2 h-2 bg-current rounded-full"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [1, 0.5, 1],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.4,
        }}
      />
    </div>
  );
}

export function FullPageLoader() {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-slate-950/80 backdrop-blur-md"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="text-center space-y-6">
        <motion.div
          className="relative"
          animate={{ rotate: 360 }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          <div className="w-24 h-24 border-4 border-blue-200 dark:border-blue-900 rounded-full" />
          <div className="absolute inset-0 w-24 h-24 border-4 border-transparent border-t-blue-600 dark:border-t-blue-400 rounded-full" />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <p className="text-lg font-semibold text-slate-700 dark:text-slate-300">
            Загрузка...
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
            Подождите, выполняется обработка
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
}
