"""Тесты: immutable domain entities (frozen dataclasses)."""
import pytest
from dataclasses import FrozenInstanceError
from src.errors import (
    DomainError, VacancyError, ApiError, ModelError, ParseError,
    PipelineError, CacheError,
)


class TestFrozenErrors:
    def test_domain_error_is_frozen(self):
        err = DomainError(message="test")
        with pytest.raises(FrozenInstanceError):
            err.message = "changed"

    def test_vacancy_error_inherits_frozen(self):
        err = VacancyError(message="not found", vacancy_id="123")
        with pytest.raises(FrozenInstanceError):
            err.vacancy_id = "456"

    def test_api_error_inherits_frozen(self):
        err = ApiError(message="api fail", status_code=500, endpoint="/test")
        with pytest.raises(FrozenInstanceError):
            err.status_code = 200

    def test_pipeline_error_inherits_frozen(self):
        err = PipelineError(message="fail", stage="extract")
        with pytest.raises(FrozenInstanceError):
            err.stage = "other"

    def test_cache_error_inherits_frozen(self):
        err = CacheError(message="cache miss", cache_path="/tmp/cache")
        with pytest.raises(FrozenInstanceError):
            err.cache_path = "/new/path"

    def test_model_error_defaults(self):
        err = ModelError(message="model load failed", model_name="test_model")
        assert err.model_name == "test_model"
        assert "model load failed" in str(err)

    def test_can_still_raise(self):
        import pytest
        with pytest.raises(DomainError):
            raise DomainError(message="boom")

    def test_detail_defaults_to_empty(self):
        err = DomainError(message="test")
        assert err.detail == ""
