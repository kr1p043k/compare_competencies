import { memo } from "react";

import { VacancyCard } from "../VacancyCard";

// Строки списка не перерисовываются от ререндеров VacanciesList
// (health-poll и статусы пайплайна дёргают App сверху): React.memo обёртка.
export const MemoVacancyCard = memo(VacancyCard);
