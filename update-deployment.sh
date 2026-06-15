cd ~/competency-platform

set -e  # exit on error

echo "Starting deployment update..."

# Backup
echo "Creating backup..."
./backup.sh
docker exec -t competency-postgres pg_dump -U competency_user compare_competencies > backup_$(date +%Y%m%d_%H%M%S).sql

# Stop services
echo "Stopping services..."
docker compose down

# Backup old compose file
cp docker-compose.yml docker-compose.yml.old

# Validate new config
echo "Validating new configuration..."
docker compose config > /dev/null

# Start with new config
echo "Starting with new configuration..."
docker compose up -d --wait --timeout 60

# Check health
echo "Checking health..."
sleep 10
if curl -f http://localhost:8000/health; then
    echo "Update successful!"
    docker compose ps
else
    echo "Health check failed! Rolling back..."
    docker compose down
    mv docker-compose.yml.old docker-compose.yml
    docker compose up -d
    exit 1
fi

# Cleanup
echo "🧹 Cleaning up old images..."
docker system prune -f

echo "Update complete!"