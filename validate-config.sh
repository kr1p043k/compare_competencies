docker compose -f docker-compose.prod.yml config
docker compose -f docker-compose.prod.yml --env-file .env config --quiet