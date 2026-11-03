# Деплой на compareserver

## Как это работает

`main` → сервер, без участия GitHub Actions:

```
systemd timer (5 мин) → cc-autodeploy.sh → scripts/deploy.sh
```

`cc-autodeploy.sh` делает `git fetch`, сравнивает `origin/main` с коммитом,
записанным в `~/.cc-deployed-sha`, и если они разошлись — запускает
`scripts/deploy.sh` из **целевого** коммита. Журнал: `logs/autodeploy.log`,
`journalctl -u cc-autodeploy`.

Почему не GitHub Actions: сервер за провайдерским NAT (интерфейс
`10.131.65.30`, публичный `95.174.102.182` — общий egress хостера), входящий
SSH доступен только через Tailscale, а ключ `TAILSCALE_AUTH_KEY` для CI истёк.
GitHub-раннер получает `dial tcp 95.174.102.182:22: i/o timeout`.

## Что делает scripts/deploy.sh

1. `chown` bind-mount'ов `data/`, `logs/`, `uploads/` (контейнеры пишут туда от
   root и ломают `git reset`; разрешено узким правилом
   `/etc/sudoers.d/compare-deploy`).
2. `git fetch` + `git reset --hard origin/main`.
3. Помечает текущие образы `:previous` — это точка отката.
4. `docker compose build backend`, `alembic upgrade heads`, пересоздание backend.
5. Ждёт `/health` до 5 минут; если не поднялся — логи и откат на `:previous`.
6. Сборка и пересоздание frontend (бандл собирается внутри образа).
7. Смоук: `/api/academic/programs` без токена должен отдавать 401.

## Установка (один раз)

```bash
sudo install -m 755 deploy/cc-autodeploy.sh /home/admin_server/bin/cc-autodeploy.sh
sudo install -m 644 deploy/cc-autodeploy.service deploy/cc-autodeploy.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cc-autodeploy.timer
```

Правило sudoers (только этот chown, без пароля):

```
admin_server ALL=(root) NOPASSWD: /usr/bin/chown -R admin_server\:admin_server \
  /home/admin_server/compare_competencies/data \
  /home/admin_server/compare_competencies/logs \
  /home/admin_server/compare_competencies/uploads
```

## Ручной запуск

```bash
sudo systemctl start cc-autodeploy.service   # один прогон вне расписания
bash scripts/deploy.sh                       # деплой текущего origin/main сразу
```

## Откат

```bash
cd /home/admin_server/compare_competencies
git reset --hard <коммит> && git rev-parse HEAD > ~/.cc-deployed-sha
docker tag compare_competencies-backend:latest.previous compare_competencies-backend:latest
docker tag compare_competencies-frontend:latest.previous compare_competencies-frontend:latest
docker compose up -d backend frontend --force-recreate
```

## Ограничения

- Деплой стартует не мгновенно после пуша, а в пределах ~5 минут.
- Сборка идёт на самом сервере, поэтому пуш в main = нагрузка на прод (сборка
  ограничена `Nice=10`, но диск и CPU она занимает).
- Таймер не проверяет CI: красный CI деплой не остановит.
