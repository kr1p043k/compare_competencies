#!/bin/bash
set -e

echo "=== Starting deployment ==="
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

backup() {
  echo "=== Backup ==="
  mkdir -p backups

  for volume in $(docker volume ls -q | grep competency-); do
    echo "Backing up volume: $volume"
    docker run --rm \
      -v $volume:/data \
      -v "$(pwd)/backups:/backup" \
      alpine tar czf /backup/${volume}_${BACKUP_DATE}.tar.gz -C /data .
  done

  echo "Backing up database..."
  docker exec -t competency-postgres pg_dump -U postgres compare_competencies \
    > backups/db_${BACKUP_DATE}.sql

  echo "Backup complete: ${BACKUP_DATE}"
}

deploy() {
  backup

  echo "=== Deploy ==="

  cp docker-compose.yml docker-compose.yml.old

  echo "Pulling latest code..."
  git pull origin main

  echo "Stopping services..."
  docker compose down

  echo "Validating config..."
  docker compose config > /dev/null

  echo "Starting services..."
  docker compose up -d --build

  echo "Health check..."
  sleep 10
  if curl -sf http://localhost:8000/health; then
    echo "Deploy successful!"
    docker compose ps
  else
    echo "Health check failed! Rolling back..."
    docker compose down
    mv docker-compose.yml.old docker-compose.yml
    docker compose up -d
    exit 1
  fi

  echo "Cleanup..."
  docker image prune -f

  echo "=== Deploy complete ==="
}

case "${1:-deploy}" in
  backup) backup ;;
  deploy) deploy ;;
esac
