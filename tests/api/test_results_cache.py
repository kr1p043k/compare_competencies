"""Кэш result-файлов: свежесть, изоляция копий, поведение на битом файле.

Кэш существует потому, что `result/*/_full_*.json` перечитывались на каждый
запрос. Ключ - (path, mtime_ns, size): правка файла обязана подхватываться без
перезапуска, а вызывающий код получает копию, чтобы правки одного запроса не
утекали в ответы другого пользователя.
"""

import json
import os

import pytest

from src.api_pkg.routers import results


@pytest.fixture(autouse=True)
def _clean_cache():
    results._reset_results_cache()
    yield
    results._reset_results_cache()


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_reads_and_parses(tmp_path):
    f = _write(tmp_path / "a.json", {"evaluations": {"x": 1}})
    assert results._read_json(f) == {"evaluations": {"x": 1}}


def test_returns_independent_copies(tmp_path):
    """Мутация результата не должна попадать в кэш и в следующий вызов."""
    f = _write(tmp_path / "b.json", {"evaluations": {"x": 1}})

    first = results._read_json(f)
    first["evaluations"]["x"] = 999

    second = results._read_json(f)
    assert second["evaluations"]["x"] == 1, "правка одного ответа протекла в кэш"


def test_cache_hit_avoids_reparse(tmp_path, monkeypatch):
    """Второй вызов при неизменном файле не должен открывать его заново."""
    f = _write(tmp_path / "c.json", {"v": 1})
    results._read_json(f)

    calls: list[str] = []
    real_open = open

    def spy(path, *a, **kw):
        calls.append(str(path))
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", spy)
    results._read_json(f)
    assert not calls, f"файл перечитан при неизменном mtime: {calls}"


def test_changed_file_is_reread(tmp_path):
    """Правка файла подхватывается без перезапуска процесса.

    mtime сдвигаем явно: на NTFS разрешение часов грубее, чем интервал между
    двумя записями в тесте, и без os.utime подпись кэша осталась бы прежней.
    """
    f = _write(tmp_path / "d.json", {"v": 1})
    assert results._read_json(f)["v"] == 1

    _write(f, {"v": 2})
    stat = f.stat()
    os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert results._read_json(f)["v"] == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        results._read_json(tmp_path / "nope.json")


def test_broken_json_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        results._read_json(f)


def test_broken_file_does_not_poison_cache(tmp_path):
    """После ошибки исправленный файл должен читаться нормально."""
    f = tmp_path / "fix.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        results._read_json(f)

    _write(f, {"ok": True})
    assert results._read_json(f) == {"ok": True}
