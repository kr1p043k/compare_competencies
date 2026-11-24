"""Барьер: Dockerfile запускает приложение от непривилегированного пользователя
(BACKLOG 6.7.2, Zone D - Docker hardening).

Офлайн-тест: Dockerfile парсится как текст, ``docker build`` не запускается и
сеть не нужна. Форматирование (перенос строк через `\\``, лишние
пробелы, регистр инструкции, ``user:group``) проверки не ломает.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"

ROOT_IDENTITIES = {"root", "0"}
#: пути, в которые приложение пишет в рантайме, обязаны быть chown'уты
WRITABLE_ENV_KEYS = ("HOME", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def logical_lines(text: str) -> list[str]:
    """Дockerfile -> список логических инструкций.

    Склеивает continuation-строки (оканчивающиеся на ``\\``), выкидывает
    пустые строки и комментарии.
    """
    result: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        result.append((buf + line).strip())
        buf = ""
    if buf.strip():
        result.append(buf.strip())
    return result


def instructions(text: str) -> list[tuple[str, str]]:
    """Логические строки -> [(ИНСТРУКЦИЯ, аргументы)]; unknown-строки пропускаются."""
    parsed: list[tuple[str, str]] = []
    for line in logical_lines(text):
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s+(.*)$", line)
        if match:
            parsed.append((match.group(1).upper(), match.group(2).strip()))
    return parsed


def user_values(text: str) -> list[str]:
    return [args for instr, args in instructions(text) if instr == "USER"]


def chown_targets(text: str) -> list[str]:
    """Все пути-мишени ``chown`` из RUN-инструкций (с учётом -R и флагов)."""
    targets: list[str] = []
    for instr, args in instructions(text):
        if instr != "RUN":
            continue
        for chunk in re.split(r"&&|\|;", args):
            if not re.search(r"\bchown\b", chunk):
                continue
            for token in chunk.split():
                token = token.strip("'\"")
                if token.startswith("/"):
                    targets.append(token.rstrip(";/"))
    return targets


def env_assignments(text: str) -> dict[str, str]:
    """Собирает ENV-присваивания (в т.ч. многострочные ``KEY=value \\``)."""
    values: dict[str, str] = {}
    for instr, args in instructions(text):
        if instr != "ENV":
            continue
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", args):
            values[key] = value
    return values


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8", errors="ignore")


def test_dockerfile_present_and_nonempty() -> None:
    assert DOCKERFILE.is_file(), f"нет Dockerfile: {DOCKERFILE}"
    assert dockerfile_text().strip(), "Dockerfile пуст"


def test_has_non_root_user_directive() -> None:
    """(a) есть USER с ненулевым, не-root значением."""
    users = user_values(dockerfile_text())
    assert users, "в Dockerfile нет ни одной инструкции USER - контейнер running от root"
    non_root = [u for u in users if u.split(":")[0].strip() not in ROOT_IDENTITIES]
    assert non_root, f"USER объявляет только root: {users}"


def test_healthcheck_present() -> None:
    """(b) HEALTHCHECK сохранён (не ослаблен при харденинге)."""
    text = dockerfile_text()
    assert any(instr == "HEALTHCHECK" for instr, _ in instructions(text)), "в Dockerfile нет HEALTHCHECK"
    assert "NONE" not in [
        args.split()[0].upper() for instr, args in instructions(text) if instr == "HEALTHCHECK"
    ], "HEALTHCHECK отключён (HEALTHCHECK NONE)"


def test_no_user_root_directive() -> None:
    """(c) нет ``USER root`` / ``USER 0`` (в т.ч. с группой)."""
    for value in user_values(dockerfile_text()):
        user = value.split(":")[0].strip()
        assert user not in ROOT_IDENTITIES, f"найден USER root: {value!r}"


def test_ownership_granted_before_switch_user() -> None:
    """(d) chown выполняется ДО переключения USER (иначе appuser не получит доступ)."""
    text = dockerfile_text()
    parsed = instructions(text)
    chown_idx = [i for i, (instr, args) in enumerate(parsed) if instr == "RUN" and re.search(r"\bchown\b", args)]
    assert chown_idx, "в Dockerfile нет chown: каталоги приложения принадлежат root"
    user_idx = [i for i, (instr, _) in enumerate(parsed) if instr == "USER"]
    assert user_idx, "нет USER - проверка порядка невозможна"
    assert min(chown_idx) < max(user_idx), "chown идёт после USER - права не применятся"


def test_app_dir_ownership_transferred() -> None:
    """Рабочий каталог /app (данные, логи, uploads) передан appuser."""
    targets = chown_targets(dockerfile_text())
    assert any(t == "/app" or t.startswith("/app/") for t in targets), f"/app не в chown-мишенях: {targets}"


def test_writable_cache_dirs_ownership_transferred() -> None:
    """Дом/HF-кэш вне /app тоже покрыты chown (иначе запись упадёт с EACCES)."""
    text = dockerfile_text()
    env = env_assignments(text)
    targets = chown_targets(text)
    assert targets, "нет chown-мишеней"
    for key in WRITABLE_ENV_KEYS:
        value = env.get(key)
        if not value:
            continue
        path = value.rstrip("/")
        if path.startswith("/app"):
            continue  # покрыто проверкой /app
        covered = any(path == t or path.startswith(t + "/") or t.startswith(path) for t in targets)
        assert covered, f"{key}={value} вне chown-мишеней {targets}: кэш недоступен непривилегированному user"


def test_runtime_paths_not_under_root_home() -> None:
    """HF/домашние пути не ссылаются на /root: к ним нет доступа у appuser."""
    env = env_assignments(dockerfile_text())
    for key, value in env.items():
        if key in WRITABLE_ENV_KEYS:
            assert not value.startswith("/root"), f"{key}={value} лежит в /root (недоступно appuser)"


def test_hf_cache_env_is_set_before_model_download() -> None:
    """Кэш задан ДО скачивания модели (иначе вес уедет в дефолт /root/.cache).

    Это и был P0-1: модель качается на этапе сборки от root, поэтому при
    отсутствии HF_HOME она попадает в /root/.cache/huggingface, а рантайм под
    appuser её не видит и падает/качает заново.
    """
    parsed = instructions(dockerfile_text())
    env_idx = [
        i
        for i, (instr, args) in enumerate(parsed)
        if instr == "ENV" and re.search(r"\bHF_HOME\s*=", args)
    ]
    download_idx = [
        i
        for i, (instr, args) in enumerate(parsed)
        if instr == "RUN" and "SentenceTransformer" in args
    ]
    assert env_idx, "в Dockerfile нет ENV HF_HOME"
    assert download_idx, "не найден шаг загрузки SentenceTransformer"
    assert min(env_idx) < min(download_idx), (
        "ENV HF_HOME объявлен после загрузки модели: вес осядет в /root/.cache, "
        "недоступном appuser"
    )



def test_healthcheck_and_cmd_need_no_root() -> None:
    """HEALTHCHECK/CMD не требуют привилегий: нет bind на порт <1024, нет ``--user 0``."""
    text = dockerfile_text()
    parsed = instructions(text)
    for instr, args in parsed:
        assert "--privileged" not in args, f"{instr} добавляет --privileged"
        if instr in {"CMD", "ENTRYPOINT", "HEALTHCHECK"}:
            port = re.search(r"--port[= ](\d+)", args)
            if port:
                assert int(port.group(1)) >= 1024, f"привязка к привилегированному порту {port.group(1)}"


def test_no_secrets_in_dockerfile_layers() -> None:
    """Харденинг не притащил секретов в образ: нет ENV/ARG с токенами, нет COPY .env."""
    text = dockerfile_text()
    for line in logical_lines(text):
        upper = line.upper()
        if upper.startswith(("ENV ", "ARG ")):
            for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", line):
                looks_secret = re.search(r"(TOKEN|PASSWORD|SECRET|KEY)$", key.upper())
                assert not (looks_secret and value.strip("\"'")), f"секрет в слое: {key}=..."
        if upper.startswith(("COPY ", "ADD ")):
            assert ".env" not in line and ".secrets" not in line, f"Dockerfile копирует секреты: {line}"
