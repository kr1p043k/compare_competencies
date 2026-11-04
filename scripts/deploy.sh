#!/usr/bin/env bash
# Деплой на прод-сервер.
#
# Запускается SSH-шагом GitHub Actions (.github/workflows/deploy.yml) и вручную:
#   bash scripts/deploy.sh
#
# Почему скрипт, а не длинный inline-script в workflow: шаги деплоя должны
# версионироваться вместе с кодом и проверяться локально. Workflow остаётся
# тонкой обёрткой «подключиться и запустить».
set -euo pipefail

APP_DIR="${APP_DIR:-/home/admin_server/compare_competencies}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:8080/}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

cd "$APP_DIR"

log() { printf '[deploy %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Контейнеры пишут в bind-mount'ы от root, поэтому часть tracked-файлов в data/
# и logs/ периодически оказывается root-owned и ломает `git reset --hard`.
# Правило в /etc/sudoers.d/compare-deploy разрешает ровно этот chown без пароля.
if ! sudo -n /usr/bin/chown -R admin_server:admin_server \
      "$APP_DIR/data" "$APP_DIR/logs" "$APP_DIR/uploads" 2>/dev/null; then
  log "WARN: passwordless chown недоступен — если reset упадёт, проверьте права"
fi

log "fetch origin/main"
git fetch origin main --quiet
PREV=$(git rev-parse --short HEAD)
git reset --hard origin/main
log "code: $PREV -> $(git rev-parse --short HEAD)"

BACKEND_IMAGE="compare_competencies-backend:latest"
FRONTEND_IMAGE="compare_competencies-frontend:latest"
BACKEND_PREV="${BACKEND_IMAGE}.previous"
FRONTEND_PREV="${FRONTEND_IMAGE}.previous"

# Образ на откат обновляем только если текущий контейнер жив: иначе можно
# затереть последний рабочий образ уже сломанным.
if curl -sf "$HEALTH_URL" >/dev/null; then
  docker tag "$BACKEND_IMAGE" "$BACKEND_PREV" 2>/dev/null || true
  docker tag "$FRONTEND_IMAGE" "$FRONTEND_PREV" 2>/dev/null || true
  log "rollback images: $BACKEND_PREV / $FRONTEND_PREV"
fi

log "build backend"
docker compose build backend

log "migrations (alembic upgrade heads)"
docker compose run --rm --no-deps --entrypoint alembic backend upgrade heads

log "recreate backend"
docker compose up -d backend --force-recreate

wait_health() {
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf "$HEALTH_URL" >/dev/null; then return 0; fi
    sleep 5
  done
  return 1
}

if ! wait_health; then
  log "ERROR: backend не поднялся за ${HEALTH_TIMEOUT}s"
  docker logs --tail 100 compare-competencies-api || true
  if docker image inspect "$BACKEND_PREV" >/dev/null 2>&1; then
    log "откат backend на предыдущий образ"
    docker tag "$BACKEND_PREV" "$BACKEND_IMAGE"
    docker compose up -d backend --force-recreate || true
  fi
  exit 1
fi

log "build frontend"
docker compose build frontend
docker compose up -d frontend --force-recreate
sleep 3
if ! curl -sf -o /dev/null "$FRONTEND_URL"; then
  log "ERROR: frontend не отвечает на $FRONTEND_URL"
  if docker image inspect "$FRONTEND_PREV" >/dev/null 2>&1; then
    docker tag "$FRONTEND_PREV" "$FRONTEND_IMAGE"
    docker compose up -d frontend --force-recreate || true
  fi
  exit 1
fi

# Смоук: маршруты новой интеграции должны существовать и требовать токен.
code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/academic/programs || true)
if [ "$code" != "401" ]; then
  log "WARN: /api/academic/programs вернул $code, ожидали 401 без токена"
fi

docker image prune -f >/dev/null 2>&1 || true
docker compose ps
log "готово"
