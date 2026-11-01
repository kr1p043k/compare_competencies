# NOTES — compare_online (ветка for_Nikita)

## 2026-11-01 — роутер, инлайн-стили, parsed_skills

Отчёт по сессии (Telegram недоступен: timeout, notify.ps1 отдал ошибку).

### Сделано
- **Hash-роутер вкладок.** Вкладка хранилась в `useState("vacancies")` — deep links
  и кнопка «назад» не работали. Добавлен `frontend/src/lib/router.ts`
  (`useTabRoute`): адрес вида `#/teacher`, неизвестное значение нормализуется к
  fallback, реакция на `hashchange`. `App.tsx` переведён на хук; сигнатура
  `setActiveTab` (значение или updater) сохранена, поэтому все 10 вызовов
  `setActiveTab("data")` и т.п. работают как раньше.
  - Нюанс: при `Page.navigate` на битый хеш нормализация нужна не только на
    mount, но и в обработчике `hashchange` — иначе мусорный адрес остаётся в
    строке (поправлено).
  - Проверка: `tools/router-check.cjs` (CDP) — deep link `#/teacher` open,
    клик «Помощь» → `#/help`, неизвестный → `#/vacancies`, `history.back()` → ok.
  - Тесты: `frontend/src/lib/router.test.tsx` (7 шт.).
- **Инлайн-стили → Tailwind.** Конвертер `.tmp_styles.py` (временный, удалён)
  разбирал `style={{...}}`: статичные свойства → классы, data-driven (цвет по
  состоянию, ширина прогресса, opacity) остались в `style`.
  - Баг конвертера: свойства, записанные одной строкой (`padding: "8px 10px",
    marginBottom: 6, borderRadius: 6, fontSize: 12, ...`), не разбивались —
    поправлено вручную (`TeacherDashboard.tsx`, карточка рекомендации).
  - Итог: **0 блоков со статикой в `style`** во всех `*.tsx` (кроме `ui/`).
- **parsed_skills.** Job `populate-parsed-skills` (17:13→17:25,
  `pipeline_runs.action='reparse'`, status `completed`):
  **50849 / 56997** вакансий с навыками; `parsed_skills IS NULL` = 0.
  Остальные 6148 — пустой список в источнике (`[]`), разбирать нечего.
- **Верификация.** `tsc --noEmit` = 0 ошибок, `npm run build` ok,
  `vitest run` = **45 passed** (7 файлов), визуальный прогон 15 вкладок
  (`tools/visual-check.cjs`) — 0 проблем (шрифт/градиенты/overflow/консоль/HTTP).
- Коммит: `d76ddbfe` (for_Nikita).

### Внимание (не мои правки)
- `data/reference/krm_disciplines_09.03.02.json` — изменён в рабочей копии
  (mtime 17:04, −384 строки: выпали коды ПК-19, ПК-22, УК-4, УК-8 и др.).
  Похоже на чужие `scripts/merge_annotations_to_krm.py` / `sfu_annotations.py`
  (они перезаписывают non-clean KRM-файлы). **Не коммитил и не откатывал** —
  решить перед коммитом (возможна потеря кодов компетенций).
- Не тронуты чужие файлы: `labeling/*`, `scripts/adjudicate.py`,
  `scripts/battery*.py`, `scripts/calibrate_*.py`, `scripts/train_ltr_ranker.py`,
  `Запуск Докера.md`, `Метрики.md`, `scripts/merge_annotations_to_krm.py`.

### Инфраструктурные заметки
- `notify.ps1` лежит в `C:\forCode\notify.ps1` (в `.config\opencode\` его нет,
  путь в AGENTS.md устарел).
- Сервис PostgreSQL `postgresql-x64-17` запускается только с UAC.
- HH выдаёт один токен на приложение в час; повторный запрос к `/token` при
  живом токене → 403. `HH_ACCESS_TOKEN` в `.env` удалён (revoked).

## 2026-11-01 (вечер) — координация параллельных агентов

В репозитории работают два агента. Чтобы не мешать друг другу — разграничение
путей и правила коммита.

**Мой трек (эта сессия):**
`src/api_pkg/*`, `src/pipeline/background_collector.py`, `src/pipeline/db_writer.py`,
`src/predictors/*`, `src/parsing/*`, `src/cli/*`, `frontend/*`, `tests/*`,
`tools/*`, `.gitignore`, `NOTES.md`.

**Трек второго агента (не трогаю):**
`data/reference/krm_disciplines_*.json`, `data/reference/curriculum_years_*.json`,
`scripts/merge_annotations_to_krm.py`, `scripts/curriculum_years.py`,
`src/pipeline/teacher_analysis_runner.py`, `labeling/*`,
`scripts/*.py` (LTR/ensemble/калибровка), `Запуск Докера.md`, `Метрики.md`.

**Правила (согласовано через deepseek-advisor):**
1. Никогда `git add -A` / `git add .` — только явные пути
   (в ветке есть плагин `auto-commit`, риск утащить чужое полуготовое).
2. Перед коммитом — `git status` и `git diff --cached --name-only`.
3. Чужие незакоммиченные файлы не редактировать; сообщать о них в этот раздел.

Сделано (эта сессия): роутер вкладок, инлайн-стили, parsed_skills
(50849/56997), прогнозы на trend_snapshots, AbortError, автопарсинг навыков
при записи в БД (`save_vacancies_batch`), старт API без блокировки вармапом,
зафиксированы labeling/скрипты и правка KRM.
