# Задача на разметку: pairs «навык РПД - навык рынка» для matcher'а

Backlog 1.5. Документ задаёт цель, формат данных, процесс и правила
качества разметки, результаты которой напрямую питают пайплайн
`src/analyzers/skill_matcher.py` и методику `docs/methodology.md`.

## 1. Цель

1. **Raise matching precision.** Каскад matcher'а (exact -> fuzzy ->
   mapped -> lemma -> anchor -> semantic, `skill_matcher.py:565-668`)
   принимает решения на порогах и allowlist'ах (adaptive threshold,
   `skill_matcher.py:173-193`; fringe-фильтр `MARKET_MIN_FREQ = 5`,
   `skill_matcher.py:33`). Размеченные пары дают ground truth, по которому
   видно, какие слои дают false positive и false negative.
2. **Provide a holdout.** В `docs/methodology.md` §2 прямо записано:
   «размеченных человеком пар "профиль-навык" нет», поэтому LTR учится на
   суррогатном таргете (`methodology.md:56-73`). Эта разметка - первый
   human-labelled набор для пары «навык из РПД - канонический навык рынка»,
   годящийся и как holdout для matcher'а, и как seed для настоящей LTR.

## 2. Что размечаем

**Тип пары: skill (текст из РПД) <-> canonical market skill (из parsed_skills
вакансий).** Это ровно та пара, на которой работает
`SkillMatcher.match(skill_name)` (`skill_matcher.py:556-558`): вход - строка
навыка РПД, выход - `(market_skill | None, тип стадии, уверенность)`.

Разметчик отвечает на один вопрос:

> действительно ли навык рынка X - то же самое (или честно применимая
> замена), что и формулировка Y из рабочей программы?

- **1** - да: тот же инструмент/технология/компетенция
  («владеть Python» -> `python`, «Базы данных (SQL)» -> `sql`);
- **0** - нет: разный смысл, слишком широко или парсит-мусор
  («программирование ПЛК» -> `plc programming` - 1, а «программирование»
  -> `python` - 0; «коммуникабельность» -> любой тех-навык - 0).

Мягкую трёхуровневую шкалу (2/1/0) роли-навык из
`labeling/guidelines.md` здесь **не** используем: matcher бинарен (сработал
или нет), поэтому и разметка бинарная 0/1.

## 3. Вход: откуда берём пары (формат экспорта)

Market-навыки берутся из колонки `vacancies.parsed_skills` (JSONB-массив
строк whitelist-канонов, `src/models/krm_models.py:46`; заполняется
`python -m src.cli populate-parsed-skills`, `src/cli/populate_parsed_skills.py`).
Навыки РПД - из распарсенных компетенций профилей base/dc/top_dc.

Экспорт candidate pairs (SQL для частотного пула рынка, тот же паттерн, что
в `src/cli/backfill_market_snapshots.py:37`):

```sql
-- 1) Частоты канонических навыков рынка (вход в set_market matcher'а)
SELECT ps AS skill, COUNT(*) AS freq
FROM vacancies,
     CROSS JOIN LATERAL jsonb_array_elements_text(v.parsed_skills::jsonb) AS ps
WHERE v.parsed_skills IS NOT NULL AND v.parsed_skills::text != '[]'
GROUP BY ps
HAVING COUNT(*) >= 5          -- MARKET_MIN_FREQ, fringe не размечаем
ORDER BY freq DESC;

-- 2) Тексты навыков РПД: выгрузка компетенций по всем трём профилям
--    (таблица компетенций + linked skills, order by profession_code)
```

Из выгрузок собирается CSV-пачка `labeling/matcher_pairs_round1.csv`:

- **POS-candidates:** строки РПД, на которых matcher уже вернул match -
  размечаем, подтверждая (`label=1`) или опрокидывая (`label=0`).
  Сформировать прогоном `SkillMatcher.match()` по всем навыкам РПД и
  выгрузкой непустых результатов вместе со стадией (`match_type`:
  exact/fuzzy/mapped/lemma/anchor/semantic) - она попадёт в служебную
  колонку `stage`, разметчик её не видит.
- **NEG-candidates:** строки РПД, вернувшие `no_match`, но с ближайшим
  рыночным кандидатом из `SkillMatcher.closest_hint()`
  (`skill_matcher.py:477-495`) - тут решаем, честный это пропуск или
  false negative semantic-слоя.
- **Hard negatives:** частотные пары, где lexical overlap есть, а смысла
  нет (примеры из кода: «отчетная документация» vs «документация», v36,
  `skill_matcher.py:29-32`; «java» vs «javascript»).
- Баланс: ~50% 1 / ~50% 0 внутри раунда, иначе baseline-большинство
  обнуляет метрики. Объём раунда 1: 400-600 пар, 3 разметчика.

## 4. Выход: формат разметки (CSV)

Один файл на разметчика, UTF-8 с BOM, разделитель `;` (открывается в Excel,
как `labeling/pairs_excel.csv`):

| id | text | canonical_skill | label |
|----|------|-----------------|-------|
| m2r-000137 | владение Python 3 | python | 1 |
| m2r-000245 | программирование логических контроллеров | python | 0 |
| m2r-000311 | работа с HTML и CSS | html | 1 |

- `id` - стабильный хэш пары `sha1(rpd_skill_text + "\x1f" + canonical_skill)[:12]`;
  один и тот же id у всех разметчиков и во всех раундах;
- `text` - исходная формулировка навыка из РПД (без нормализации, как в
  программе);
- `canonical_skill` - навык рынка ровно из `parsed_skills` (нижним регистром);
- `label` - только `0` или `1`, пустых значений не бывает: сомнение
  помечается колонкой `flag=true` и уходит на арбитраж (§6).

Служебные колонки `stage`, `freq` экспортер добавляет, разметчик их не
трогает; при склейке колонок `label` от трёх разметчиков они сохраняются.

## 5. UI и процесс

Процесс повторяет существующий раунд `labeling/` (голосования
`votes_e*.csv`, арбитраж `adjudicate.py`, scoring `score.py`):

1. Экспортер (скрипт §3) нарезает пачку 400-600 пар, перемешивает,
   дублирует 10% пар как скрытые реплики (для замера intra-annotator
   consistency), раскладывает по трём файлам `votes_m1_<role>.csv`.
2. Разметчик открывает CSV в Excel/Google Sheets (проверенный путь -
   `labeling/pairs_excel.csv`) либо в streamlit-просмотрщике поверх того же
   CSV, если пачка > 1000 пар. Пустого UI-фреймворка ради 1.5 не строим.
3. Раунд 1 - независимо, 3 разметчика, все судят все пары. Сессии по
   60-90 минут, не больше одной за день (усталость шумит, см. правило
   времени в `labeling/guidelines.md:23-24`).
4. Раунд 2 - только disagreements: пары, где голоса разошлись, выгружаются
   заново и переразмечаются (паттерн `labeling/round_2.csv`).

## 6. Правила качества

- **Inter-annotator agreement:** Fleiss' kappa по трём колонкам, тот же
  инструмент `labeling/score.py`. Порог приличия как в
  `docs/methodology.md:114`: **kappa >= 0.4**; ниже - уточняем шкалу,
  разбираем систематические расхождения (обычно это «широкий навык РПД vs
  узкий рынок» и парсинг-мусор) и переразмечаем спорное.
- Majority = итоговый `label`; ties (2:1 оставляем большинство, 1:1:1 или
  2/1 с flag) - на арбитраж старшему преподавателю (`adjudicate.py`).
- Скрытые реплики: разметчик, перекомдивший >= 20% собственных повторов,
  переделывает раунд.
- Gold = пары с unanimous 3/3; спорные после арбитража помечаются
  `resolved=1` и в precision-метрику не входят (входят только в recall-пул).

## 7. Как результат попадает в docs/methodology.md

1. **F1-cascade (Exp7, `methodology.md:116-122`).** По gold-разметке
   считается precision/recall/F1 отдельно по каждой стадии каскада
   (`stage`-колонка §3): какие стадии дают FP (fuzzy-ловушки, semantic при
   низком пороге), какие - FN. Это заменяет синтетические 800 пар KSA.
2. **Пороги semantic-слоя.** `adaptive_semantic_threshold`
   (`skill_matcher.py:173-193`, base 0.55, clamp [0.45, 0.75]) калибруется
   на ROC по human-labels вместо пар Exp2 (163 пары,
   `methodology.md:121`); новый пик F1 фиксируется в таблице §3.1.
3. **LTR (§2 `methodology.md:56-73`).** Бинарные пары - первый шаг от
   surrogate-таргета к настоящему pairwise/pointwise LTR: human-label
   заменяет `relevance` в `target = 0.6 x relevance + 0.4 x base_target`
   для тех пар, где он есть; holdout-сплит (seed 42) прогоняется через
   `scripts/eval_frozen.py` и даёт новый ключ `human_ndcg_at_10` в
   `data/reports/frozen_eval.json` - по нему, а не по хардкоду, выбирается
   prod-модель (`methodology.md:74-102`).
4. **Регресс-гейт.** Перед каждым изменением порогов/allowlist'ов matcher'а
   прогон `eval_frozen.py --check` (`methodology.md:104-109`): падение
   precision на human-holdout больше допуска - изменение не мержим.

## 8. Definition of done (раунд 1)

- `labeling/matcher_pairs_round1.csv`: >= 400 пар, баланс ~50/50,
  все id уникальны, `canonical_skill` встречается в частотном пуле §3;
- три колонки голосов + majority + kappa >= 0.4 в отчёте `score.py`;
- disagreements второго раунда закрыты арбитражем;
- таблица per-stage precision/recall приложена к §3.1 `methodology.md`.
