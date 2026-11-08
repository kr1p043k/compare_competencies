#!/bin/bash
# Ежедневный бэкап БД compare_competencies (запускается cc-backup.timer).
#
# Дамп делается на том же диске, что и БД, поэтому он спасает от
# «случайно дропнули таблицу / кривая миграция», но не от потери диска:
# для оффсайт-копии нужно периодически забирать backups/ по rsync/scp.
set -uo pipefail

APP_DIR=/home/admin_server/compare_competencies
BACKUP_DIR="$APP_DIR/backups"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

OUT="$BACKUP_DIR/db_${STAMP}.dump"
if docker exec competency-postgres pg_dump -U postgres -Fc compare_competencies > "$OUT"; then
  echo "$(date -u +%FT%TZ) ok: db_${STAMP}.dump ($(du -h "$OUT" | cut -f1))"
else
  echo "$(date -u +%FT%TZ) FAIL: pg_dump, exit $?" >&2
  rm -f "$OUT"
  exit 1
fi

# Ротация: хранить 14 последних db_*.dump (деплой тоже кладёт сюда).
ls -1t "$BACKUP_DIR"/db_*.dump 2>/dev/null | tail -n +15 | xargs -r rm -f
