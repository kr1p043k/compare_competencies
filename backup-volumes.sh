BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
for volume in $(docker volume ls -q | grep competency-); do
  docker run --rm -v $volume:/data -v ./backups:/backup alpine \
    tar czf /backup/${volume}_${BACKUP_DATE}.tar.gz -C /data .
done