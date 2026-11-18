# Retention policy

Что и как долго хранится, и чем это чистится. Реализация: `src/cli/retention.py`
(CLI-команда `retention`), dry-run по умолчанию.

## Политика

| Сущность | Срок | Условие удаления | Флаг |
|---|---|---|---|
| `vacancies` (снапшоты) | 730 дней | старше срока | `--vacancies-days` |
| логи (`request_logs`) | 90 дней | старше срока | `--logs-days` |
| `sessions` (закрытые) | 30 дней | `logged_out_at IS NOT NULL` | `--session-days` |
| `pipeline_runs` (завершённые) | 365 дней | `status in (completed, failed, cancelled)` | `--run-days` |
| `audit_logs` | 730 дней | старше срока | `--audit-days` |

Активные сессии (`logged_out_at IS NULL`) и запущенные прогоны (`status='started'`)
**не удаляются** никогда: их защищает тот же слой, что и InProgress-кеш.

## Как запускать

```bash
python -m src.cli retention                 # dry-run: только печатает план
python -m src.cli retention --apply         # выполнить удаление
python -m src.cli retention --vacancies-days 730 --logs-days 90 --apply
python -m src.cli retention --session-days 30 --run-days 365 --apply
```

## Планирование (enforcement)

Политика должна исполняться сама, а не «когда вспомнят». Ежедневно в 04:00
(с разбросом до 10 минут) чистку запускает systemd-таймер `cc-retention.timer`,
который вызывает `cc-retention.sh` -> `docker exec compare-competencies-api
python -m src.cli retention --apply`.

Установка (один раз на сервере):

```bash
sudo cp deploy/cc-retention.sh /home/admin_server/bin/cc-retention.sh
sudo chmod +x /home/admin_server/bin/cc-retention.sh
sudo cp deploy/cc-retention.service deploy/cc-retention.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cc-retention.timer
systemctl list-timers cc-retention.timer          # следующий запуск
journalctl -u cc-retention.service -n 50 --no-pager   # что удалено в последний раз
```

`Persistent=true` догоняет пропущенный запуск после простоя. Таймер не трогает
активные сессии и незавершённые прогоны (см. раздел «Политика»).

## Заметки

- По умолчанию режим dry-run: ничего не удаляется без явного `--apply`.
- Доменные кэши (`sessions`, `pipeline_runs`) чистятся здесь; рабочие данные
  (`competencies`, `skills`, KRM) retention не трогает.
- Владение: Зона D (infra/DB). Изменения политики - через ревью (этот документ
  и `src/cli/retention.py` держать синхронными).
