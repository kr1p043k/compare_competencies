/**
 * Скелетонные заглушки для ленивых (React.lazy) вкладок.
 *
 * Пока чанк вкладки грузится, Suspense показывает одну из этих заглушек.
 * Форма повторяет будущую раскладку (заголовок + список или карточки
 * дашборда), чтобы переключение вкладок не «прыгало» и не проваливалось
 * в пустоту. Для скринридеров у заглушки есть role="status" и sr-only текст.
 */
import { Skeleton } from "../../../design";

type TabSkeletonProps = {
  /** Озвучивается скринридером (sr-only): что именно грузится. */
  label: string;
};

/** Каркас списочной вкладки: заголовок + строки карточек (публикации, логи…). */
export function ListSkeleton({ label }: TabSkeletonProps) {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-8 w-72" />
      <div className="space-y-3">
        <Skeleton className="h-5 w-full max-w-xl" />
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-20 w-full rounded-xl" />
      </div>
    </div>
  );
}

/** Каркас вкладки-дашборда: строка статистики + два крупных блока (графики/таблицы). */
export function DashboardSkeleton({ label }: TabSkeletonProps) {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-8 w-72" />
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24 w-full rounded-lg" />
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    </div>
  );
}
