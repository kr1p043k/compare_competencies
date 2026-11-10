"""Гейт полноты audit_logs: мутирующий роут обязан писать аудит.

Статическая проверка (AST): для каждого обработчика POST/PUT/PATCH/DELETE в
src/api_pkg/routers требуется вызов `audit_action(...)`. Это «храповик»: новые
эндпоинты не могут появиться без записи в audit_logs, а старые пробелы явно
перечислены ниже с причиной. Когда пробел закрывают - строку убирают отсюда.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROUTERS_DIR = Path(__file__).resolve().parent.parent / "src" / "api_pkg" / "routers"
MUTATING = {"post", "put", "patch", "delete"}

# (файл, функция) -> почему аудит не требуется (или отложен).
KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("academic.py", "sso_login"): "вход через хаб: логируется через user_logged_in",
    ("academic.py", "academic_get_competencies"): "чтение (POST ради тела запроса)",
    ("academic.py", "academic_analyze_gap"): "аналитика без изменения данных",
    ("academic.py", "academic_analyze_gap_local"): "аналитика без изменения данных",
    ("admin.py", "frontend_log"): "приём клиентских логов, а не привилегированное действие",
    ("auth.py", "login"): "вход логируется отдельно (user_logged_in)",
    ("health.py", "write_log"): "приём логов фронтенда",
    ("krm_teacher.py", "krm_teacher_gap"): "аналитика без изменения данных",
    ("llm.py", "llm_chat"): "диалог с моделью, без изменения данных",
    ("pipeline.py", "pipeline_rebuild"): "фоновый прогон, статус пишется в pipeline_runs",
    ("pipeline.py", "refresh_cache"): "сброс кэша, обратимо и без ПДн",
    ("pipeline.py", "reload_api"): "TODO: привилегированное действие, стоит добавить аудит",
    ("pipeline.py", "run_pipeline_action_sync"): "фоновый прогон, статус в pipeline_runs",
    ("profiles.py", "create_custom_profile"): "TODO: изменение пользовательских данных",
    ("profiles.py", "patch_self_profile"): "TODO: изменение пользовательских данных",
    ("publications.py", "start_deep_search"): "фоновый поиск, статус в своей таблице",
    ("publications.py", "cancel_deep_search"): "отмена фонового поиска",
    ("rpd.py", "rpd_upload"): "TODO: загрузка файла пользователя",
    ("rpd.py", "rpd_collect"): "фоновый сбор, статус в pipeline_runs",
    ("rpd.py", "rpd_cancel"): "отмена фонового сбора",
    ("student.py", "student_log_action"): "клиентская телеметрия",
    ("subscriptions.py", "create_subscription"): "TODO: пользовательские данные",
    ("subscriptions.py", "delete_subscription"): "TODO: пользовательские данные",
    ("subscriptions.py", "mark_read"): "косметическая отметка прочтения",
    ("teacher.py", "run_teacher_analysis_endpoint"): "фоновый прогон, статус в pipeline_runs",
    ("zun.py", "zun_analyze"): "аналитика без изменения данных",
}


def _mutating_handlers(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_mutating = False
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "router"
                and target.attr in MUTATING
            ):
                is_mutating = True
        if not is_mutating:
            continue
        has_audit = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "audit_action"
            for n in ast.walk(node)
        )
        if not has_audit:
            found.append((node.name, node.lineno))
    return found


def test_mutating_routes_write_audit_or_are_known_gaps() -> None:
    unexpected: list[str] = []
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        for name, lineno in _mutating_handlers(path):
            if (path.name, name) not in KNOWN_GAPS:
                unexpected.append(f"{path.name}:{lineno} {name}")
    assert not unexpected, (
        "Мутирующие роуты без audit_action и без записи в KNOWN_GAPS "
        "(добавь audit_action или обоснуй исключение):\n" + "\n".join(unexpected)
    )


def test_known_gaps_are_still_real() -> None:
    """Устаревшие записи в KNOWN_GAPS (пробел уже закрыт) надо удалять."""
    still_open: set[tuple[str, str]] = set()
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        for name, _lineno in _mutating_handlers(path):
            still_open.add((path.name, name))
    stale = sorted(g for g in KNOWN_GAPS if g not in still_open)
    assert not stale, f"Записи в KNOWN_GAPS устарели (пробел закрыт): {stale}"
