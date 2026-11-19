import { motion } from "motion/react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "../../../design";

interface VacancyPaginationProps {
  currentPage: number;
  totalPages: number;
  loading: boolean;
  onPageChange: (updater: (prev: number) => number) => void;
}

/** Постраничная навигация под списком. Показывается родителем при totalPages > 1. */
export function VacancyPagination({ currentPage, totalPages, loading, onPageChange }: VacancyPaginationProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
      className="flex items-center justify-center gap-2"
    >
      <Button
        variant="secondary"
        size="sm"
        onClick={() => onPageChange((p) => Math.max(1, p - 1))}
        disabled={currentPage === 1 || loading}
        aria-label="Предыдущая страница"
        className="size-10 p-0"
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
            variant={currentPage === pageNum ? "primary" : "secondary"}
            size="sm"
            onClick={() => onPageChange(() => pageNum)}
            disabled={loading}
            aria-current={currentPage === pageNum ? "page" : undefined}
            className="size-10 p-0"
          >
            {pageNum}
          </Button>
        );
      })}

      <Button
        variant="secondary"
        size="sm"
        onClick={() => onPageChange((p) => Math.min(totalPages, p + 1))}
        disabled={currentPage === totalPages || loading}
        aria-label="Следующая страница"
        className="size-10 p-0"
      >
        <ChevronRight className="size-4" />
      </Button>
    </motion.div>
  );
}
