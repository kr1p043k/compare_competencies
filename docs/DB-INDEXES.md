# Индексы БД: acancies.parsed_skills / acancies.raw `vacancies.parsed_skills` / `vacancies.raw` (BACKLOG 6.3.1)

Дата: 2026-11-15. База: локальная `compare_competencies` (PostgreSQL 17).

## Что проверяли

Горячие запросы, фильтрующие по JSONB-полям `vacancies`:

- `parsed_skills IS NOT NULL` - `src/api_pkg/routers/forecast.py:194`, `teacher.py:1189`,
  `trends_by_profession.py:87`;
- `jsonb_array_length(parsed_skills) > 0` - `teacher.py:1189`,
  `trends_by_profession.py`;
- `parsed_skills != '[]'::jsonb` - `forecast.py:195`;
- `raw` - только читается/пишется, в фильтрах **не участвует** (по `git grep` нет
  ни `raw->>`, ни `raw->`, ни `@>`).

## Замеры

```
vacancies: total=159818, parsed_skills IS NOT NULL=151341 (94.7%), non-empty=126217
```

EXPLAIN (ANALYZE, BUFFERS):

1. `SELECT count(*) FROM vacancies WHERE parsed_skills IS NOT NULL AND
   jsonb_array_length(parsed_skills) > 0`
   → Parallel Seq Scan, `Execution Time: 153.7 ms`, `Rows Removed by Filter: 11200`.
2. `SELECT id, parsed_skills FROM vacancies WHERE parsed_skills IS NOT NULL
   AND parsed_skills != '[]'::jsonb LIMIT 100`
   → Seq Scan + Limit, `Execution Time: 1.5 ms`.

Существующие индексы на `vacancies`: `idx_vacancies_parsed_pub` (partial btree по
`published_at`), `idx_vacancies_hh_id`, `idx_vacancies_employer`,
`idx_vacancies_pipeline`, `idx_vacancies_published_nl`. Индексов по JSONB-полям нет.

## Вывод: новый индекс НЕ нужен

- `parsed_skills IS NOT NULL` неселективно (94.7% строк) - **частичный индекс по
  этому условию покрыл бы почти всю таблицу** и только добавил бы накладные
  расходы на запись без выигрыша в чтении.
- GIN по содержимому JSONB (`parsed_skills @> '["python"]'`) дал бы выигрыш только
  под containment-запросы, но таких в коде нет (см. `git grep`). Если появятся -
  добавить `CREATE INDEX ... USING gin (parsed_skills jsonb_path_ops)`.
- `vacancies.raw` не фильтруется - индекс не требуется.
- Полный `count(*)` по 160k строк за ~150 мс - приемлемо; seq scan здесь оптимален
  (читает всю таблицу всё равно). Агрегаты по объёму кэшируются на уровне кода.

## Когда пересмотреть

Добавить **partial index** `CREATE INDEX idx_vacancies_parsed_nn ON vacancies (id)
WHERE parsed_skills IS NOT NULL`, если:

- доля NULL в `parsed_skills` вырастет так, что `IS NOT NULL` станет селективным
  (ориентир: меньше ~20% строк), **и**
- появится частый запрос, который из-за этого условия сканирует всю таблицу.

Проверять тем же EXPLAIN (ANALYZE, BUFFERS) на проде, а не на глаз.

## Как воспроизвести

```sql
SELECT count(*) FILTER (WHERE parsed_skills IS NULL) AS nulls,
       count(*) AS total FROM vacancies;

EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM vacancies
WHERE parsed_skills IS NOT NULL AND jsonb_array_length(parsed_skills) > 0;
```
