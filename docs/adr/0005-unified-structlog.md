# ADR-0005: Единое логирование через setup_structlog

## Статус

accepted.

## Контекст

Логирование собиралось вразнобой: часть точек входа вызывала локальный
`logging.basicConfig(...)` (в `trends.py`, `ltr_recommendation_engine.py`,
`orchestration.py`), а `src/parsing/utils.py::setup_logging` добавлял свои
handler'ы напрямую в root-логгер. Это дублировало handler'ы, ломало
`RotatingFileHandler` и, главное, **обходило маскирование секретов**
(`SecretsMasker`), из-за чего токены/пароли могли попасть в лог.

## Решение

Единственная точка настройки - `src/logging_config.py::setup_structlog`:
structlog-конвейер + `SecretsMasker` + безопасные handler'ы (файл с ротацией +
консоль, устойчивая к кодировке). Локальные `basicConfig` убраны, дубль
`parsing/utils.setup_logging` делегирует в `setup_structlog`, а CLI-диспетчер
`src/cli/__main__.py` вызывает его один раз.

## Последствия

- Плюс: секреты маскируются во всех точках; нет дублей handler'ов.
- Плюс: единый формат/уровень через `LOG_LEVEL`.
- Минус: модуль, который печатал логи до настройки, получит только консоль
  structlog по умолчанию - точки входа обязаны вызвать `setup_structlog`.
