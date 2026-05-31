import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi import HTTPException

from src.n8n.webhooks import (
    StudentCreatedBody,
    PipelineCompletedBody,
    AlertBody,
    _verify_n8n_secret,
    webhook_student_created,
    webhook_pipeline_completed,
    webhook_alert,
    list_webhook_events,
    WEBHOOK_STORE,
)


class TestWebhookModels:
    def test_student_created_body_defaults(self):
        body = StudentCreatedBody(profile_name="test", skills=["python"])
        assert body.profile_name == "test"
        assert body.skills == ["python"]
        assert body.target_level == "middle"

    def test_pipeline_completed_body_defaults(self):
        body = PipelineCompletedBody(task_id="123", status="done")
        assert body.task_id == "123"
        assert body.status == "done"
        assert body.artifacts == {}

    def test_alert_body_defaults(self):
        body = AlertBody(type="error", message="test error")
        assert body.type == "error"
        assert body.severity == "info"
        assert body.message == "test error"
        assert body.data == {}


class TestVerifyN8NSecret:
    def test_no_secret_configured_returns_true(self):
        mock_request = MagicMock()
        mock_request.headers = {}
        with patch("src.config.N8N_WEBHOOK_SECRET", None):
            assert _verify_n8n_secret(mock_request) is True

    def test_valid_secret_returns_true(self):
        mock_request = MagicMock()
        mock_request.headers = {"X-N8N-Webhook-Secret": "my-secret"}
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "my-secret"
        with patch("src.config.N8N_WEBHOOK_SECRET", mock_secret):
            assert _verify_n8n_secret(mock_request) is True

    def test_invalid_secret_returns_false(self):
        mock_request = MagicMock()
        mock_request.headers = {"X-N8N-Webhook-Secret": "wrong"}
        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "correct"
        with patch("src.config.N8N_WEBHOOK_SECRET", mock_secret):
            assert _verify_n8n_secret(mock_request) is False


class TestWebhookStudentCreated:
    @pytest.mark.asyncio
    async def test_raises_on_invalid_secret(self):
        body = StudentCreatedBody(profile_name="test", skills=["python"])
        mock_request = MagicMock()
        mock_request.headers = {}
        with patch("src.n8n.webhooks._verify_n8n_secret", return_value=False):
            with pytest.raises(HTTPException) as exc:
                await webhook_student_created(body, mock_request)
            assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_success(self):
        body = StudentCreatedBody(profile_name="test", skills=["python"])
        mock_request = MagicMock()
        mock_request.headers = {"X-N8N-Webhook-Secret": "secret"}
        mock_path = MagicMock()
        with (
            patch("src.n8n.webhooks._verify_n8n_secret", return_value=True),
            patch("src.n8n.webhooks.WEBHOOK_STORE") as mock_store,
            patch("src.n8n.webhooks.datetime") as mock_dt,
        ):
            mock_store.mkdir.return_value = None
            mock_store.__truediv__.return_value = MagicMock()
            mock_dt.now.return_value.strftime.return_value = "20260101_120000"
            mock_dt.now.return_value.isoformat.return_value = "2026-01-01T12:00:00"

            with patch("src.n8n.webhooks.atomic_write_json") as mock_write:
                result = await webhook_student_created(body, mock_request)
                assert result["ok"] is True
                assert result["profile"] == "test"
                mock_write.assert_called_once()


class TestWebhookPipelineCompleted:
    @pytest.mark.asyncio
    async def test_success(self):
        body = PipelineCompletedBody(task_id="t1", status="ok")
        mock_request = MagicMock()
        with (
            patch("src.n8n.webhooks._verify_n8n_secret", return_value=True),
            patch("src.n8n.webhooks.WEBHOOK_STORE") as mock_store,
            patch("src.n8n.webhooks.datetime") as mock_dt,
        ):
            mock_store.mkdir.return_value = None
            mock_store.__truediv__.return_value = MagicMock()
            mock_dt.now.return_value.strftime.return_value = "20260101_120000"
            mock_dt.now.return_value.isoformat.return_value = "2026-01-01T12:00:00"
            with patch("src.n8n.webhooks.atomic_write_json"):
                result = await webhook_pipeline_completed(body, mock_request)
                assert result["ok"] is True
                assert result["task_id"] == "t1"


class TestWebhookAlert:
    @pytest.mark.asyncio
    async def test_success(self):
        body = AlertBody(type="test", message="msg")
        mock_request = MagicMock()
        with (
            patch("src.n8n.webhooks._verify_n8n_secret", return_value=True),
            patch("src.n8n.webhooks.WEBHOOK_STORE") as mock_store,
            patch("src.n8n.webhooks.datetime") as mock_dt,
        ):
            mock_store.mkdir.return_value = None
            mock_store.__truediv__.return_value = MagicMock()
            mock_dt.now.return_value.strftime.return_value = "20260101_120000"
            mock_dt.now.return_value.isoformat.return_value = "2026-01-01T12:00:00"
            with patch("src.n8n.webhooks.atomic_write_json"):
                result = await webhook_alert(body, mock_request)
                assert result["ok"] is True


class TestListWebhookEvents:
    @pytest.mark.asyncio
    async def test_returns_events(self):
        mock_request = MagicMock()
        with (
            patch("src.n8n.webhooks._verify_n8n_secret", return_value=True),
            patch("src.n8n.webhooks.WEBHOOK_STORE") as mock_store,
        ):
            mock_store.mkdir.return_value = None
            mock_store.glob.return_value = []
            result = await list_webhook_events(mock_request)
            assert "events" in result
            assert "total" in result
