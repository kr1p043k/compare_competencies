#!/usr/bin/env bash
# Автодеплой origin/main на compareserver.
#
# Почему не GitHub Actions: сервер стоит за провайдерским NAT (на интерфейсе
# 10.131.65.30, публичный 95.174.102.182 — общий egress хостера), входящий SSH
# есть только через Tailscale. GitHub-hosted раннер до него не достучится, а
# Tailscale-ключ для CI истёк. Поэтому инициатива деплоя — здесь: таймер раз в
# 5 минут сравнивает выкаченный коммит с origin/main.
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO=/home/admin_server/compare_competencies
LOG=$REPO/logs/autodeploy.log
SHA_FILE=/home/admin_server/.cc-deployed-sha
FAIL_FILE=/home/admin_server/.cc-deploy-failed
COOLDOWN=1800

mkdir -p "$(dirname "$LOG")"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# Один деплой за раз: сборка занимает минуты, второй запуск только помешает.
exec 9>/tmp/cc-autodeploy.lock
if ! flock -n 9; then
  log "skip: другой деплой уже идёт"
  exit 0
fi

cd "$REPO"

git fetch origin main --quiet || { log "ERROR: git fetch не удался"; exit 1; }
REMOTE=$(git rev-parse origin/main)
DEPLOYED=$(cat "$SHA_FILE" 2>/dev/null || echo "")

if [ "$REMOTE" = "$DEPLOYED" ]; then
  exit 0
fi

# Провал сборки не должен превращаться в бесконечный цикл пересборок.
if [ -f "$FAIL_FILE" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$FAIL_FILE") ))
  if [ "$age" -lt "$COOLDOWN" ]; then
    log "skip: предыдущая попытка упала ${age}s назад (cooldown ${COOLDOWN}s)"
    exit 0
  fi
fi

# Не трогаем рабочее дерево, если в нём есть чужие правки (ручной хотфикс).
if [ -n "$(git status --porcelain)" ]; then
  log "skip: в рабочем дереве есть незакоммиченные изменения"
  exit 0
fi

log "start: ${DEPLOYED:-none} -> $REMOTE ($(git log -1 --format=%h --abbrev=7 "$REMOTE"))"
if git show "$REMOTE:scripts/deploy.sh" > /tmp/cc-deploy.sh 2>/dev/null; then :; else
  cp "$REPO/scripts/deploy.sh" /tmp/cc-deploy.sh
fi

if bash /tmp/cc-deploy.sh >> "$LOG" 2>&1; then
  git rev-parse HEAD > "$SHA_FILE"
  rm -f "$FAIL_FILE"
  log "ok: $(git rev-parse --short HEAD)"
else
  touch "$FAIL_FILE"
  log "FAILED: см. вывод выше; повтор не раньше чем через $((COOLDOWN / 60)) мин"
  exit 1
fi
