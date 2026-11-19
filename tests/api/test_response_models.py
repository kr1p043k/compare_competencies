"""6.2.1: у ключевых роутов объявлен response_model (именованная схема, а не {})."""
from __future__ import annotations

from src.api_pkg import create_app

# путь -> имя схемы, которую роут обязан публиковать в 200-ответе OpenAPI
EXPECTED = {
    "/api/version": "ApiVersionBody",
    "/": "RootBody",
    "/api/academic/krm-competencies": "AcademicKrmCompetenciesBody",
}


def test_routes_declare_named_response_schema():
    spec = create_app().openapi()
    for path, schema_name in EXPECTED.items():
        assert path in spec["paths"], f"{path} отсутствует в OpenAPI"
        schema = spec["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{schema_name}"}, (path, schema)
        props = spec["components"]["schemas"][schema_name]["properties"]
        assert props, f"{schema_name} без полей"


def test_live_and_ready_keep_models():
    spec = create_app().openapi()
    for path in ("/live", "/api/live", "/ready", "/health"):
        if path in spec["paths"]:
            schema = spec["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
            assert "$ref" in schema, (path, schema)
