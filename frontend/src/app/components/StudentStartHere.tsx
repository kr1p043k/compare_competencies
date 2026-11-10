/**
 * «Начните здесь» для студента.
 *
 * Страница студента начиналась с пустого профиля, пустой истории и списка
 * вакансий рынка - без единой подсказки, что делать. Пользователь видел десять
 * вкладок и ноль объяснений. Этот блок отвечает на три вопроса: где я, что
 * заполнить первым, что получу в ответ.
 *
 * Если профиль уже заполнен, блок сам себя не показывает - он нужен только
 * на старте.
 */
import { useEffect, useState } from "react";
import { Button, Card, CardBody, CardHeader } from "../../design";
import { CircleCheck, Lightbulb, ArrowRight } from "lucide-react";
import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";

type SelfProfile = {
  skills?: unknown[];
  competencies?: unknown[];
  technologies?: unknown[];
  user_added?: unknown[];
};

export function StudentStartHere({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [loaded, setLoaded] = useState(false);
  const [empty, setEmpty] = useState(false);

  useEffect(() => {
    let cancelled = false;
    request<SelfProfile>("/api/profiles/self")
      .then((p) => {
        if (cancelled) return;
        const total = (p.skills?.length ?? 0)
          + (p.competencies?.length ?? 0)
          + (p.technologies?.length ?? 0)
          + (p.user_added?.length ?? 0);
        setEmpty(total === 0);
      })
      .catch((err) => reportError("StudentStartHere:profile", err))
      .finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, []);

  if (!loaded || !empty) return null;

  return (
    <Card className="border-primary/30">
      <CardHeader
        className="pb-3"
        title={
          <span className="flex items-center gap-2">
            <Lightbulb className="size-5 text-primary" />
            Начните здесь
          </span>
        }
      />
      <CardBody className="space-y-3 text-sm">
        <p className="text-muted-foreground">
          Ваш профиль пока пуст, поэтому сравнить вас с программой не с чем.
          Заполните навыки - и платформа покажет, что уже совпадает с учебным
          планом, чего не хватает и какие темы стоит подтянуть.
        </p>
        <ol className="space-y-1.5 text-foreground">
          <li className="flex gap-2">
            <CircleCheck className="size-4 mt-0.5 shrink-0 text-primary" />
            <span>Добавьте навыки и технологии, которыми уже владеете.</span>
          </li>
          <li className="flex gap-2">
            <CircleCheck className="size-4 mt-0.5 shrink-0 text-primary" />
            <span>Платформа сопоставит их с компетенциями программы ниже.</span>
          </li>
          <li className="flex gap-2">
            <CircleCheck className="size-4 mt-0.5 shrink-0 text-primary" />
            <span>Вы увидите разрывы и рекомендации, что изучать дальше.</span>
          </li>
        </ol>
        {onNavigate && (
          <Button size="sm" onClick={() => onNavigate("profile")} className="mt-1">
            Заполнить профиль
            <ArrowRight className="size-4 ml-1.5" />
          </Button>
        )}
      </CardBody>
    </Card>
  );
}

export default StudentStartHere;
