import { useEffect } from "react";

import { useTabRoute } from "../../lib/router";

import { isTabAllowedForRole } from "./tabConfig";

type SetTab = (next: string | ((current: string) => string)) => void;

/**
 * Вкладка синхронизирована с адресом: `#/teacher` и т.п.
 * (deep links + история через lib/router), плюс правила доступа:
 * при смене роли (или предпросмотра роли) уводим с недоступной вкладки
 * на безопасную «vacancies», не трогая доступную.
 *
 * `enabled` (в App - isAuth): пока пользователь не вошёл, роутер не пишет
 * адрес - экран входа не подменяет `/` на `#/vacancies`.
 */
export function useTabNavigation(
  fallback: string,
  effectiveRole: string | null,
  enabled: boolean,
): [string, SetTab] {
  const [activeTab, setActiveTab] = useTabRoute(fallback, enabled);

  useEffect(() => {
    setActiveTab((cur) => (isTabAllowedForRole(cur, effectiveRole) ? cur : fallback));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveRole]);

  return [activeTab, setActiveTab];
}
