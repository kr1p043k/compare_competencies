"""Snapshot tests for critical output structures."""

import json

from src import Err, Ok


class TestResultSnapshot:
    def test_ok_repr(self, snapshot):
        r = Ok(42)
        snapshot.assert_match(repr(r), "ok_repr.txt")

    def test_err_repr(self, snapshot):
        r = Err("fail")
        snapshot.assert_match(repr(r), "err_repr.txt")

    def test_ok_str_value(self, snapshot):
        r = Ok({"skill": "python", "weight": 0.95})
        snapshot.assert_match(json.dumps(r.unwrap(), sort_keys=True), "ok_value.json")

    def test_err_value(self, snapshot):
        r = Err({"message": "not found", "code": 404})
        snapshot.assert_match(json.dumps(r.err(), sort_keys=True), "err_value.json")


class TestConfigSnapshot:
    def test_result_module_structure(self, snapshot):
        from src import Result as R
        from src.result import Ok as OkCls, Err as ErrCls

        snapshot.assert_match(repr(OkCls), "result_Ok_class.txt")
        snapshot.assert_match(repr(ErrCls), "result_Err_class.txt")
        snapshot.assert_match(f"Result base: {R.__bases__}", "result_class_bases.txt")


class TestErrorsSnapshot:
    def test_error_hierarchy(self, snapshot):
        from src.errors import DomainError, PipelineError, SkillExtractionError

        lines = []
        errs = [DomainError, PipelineError, SkillExtractionError]
        for cls in errs:
            lines.append(f"{cls.__name__}({', '.join(f.name for f in cls.__dataclass_fields__.values())})")
        snapshot.assert_match("\n".join(lines), "error_hierarchy.txt")
