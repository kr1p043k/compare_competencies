#!/bin/bash
# Ежедневная чистка по политике docs/RETENTION.md (запускается cc-retention.timer).
#
# Пороги по умолчанию совпадают с DEFAULT_*_DAYS в src/cli/retention.py и с
# таблицей в docs/RETENTION.md. Команда идёт в контейнер backend (там есть
# python + зависимости и тот же DATABASE_URL), dry-run не нужен: это боевой
# таймер, поэтому запускаем с --apply и пишем в journal, что удалено.
set -uo pipefail

CONTAINER=compare-competencies-api

if docker exec "$CONTAINER" python -m src.cli retention --apply; then
  echo "$(date -u +%FT%TZ) ok: retention applied"
else
  echo "$(date -u +%FT%TZ) FAIL: retention, exit $?" >&2
  exit 1
fi
