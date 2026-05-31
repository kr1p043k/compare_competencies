import pytest
from src.api_pkg.n8n import N8N_ENDPOINTS, N8N_WORKFLOWS, N8N_WEBHOOKS


class TestN8NEndpoints:
    def test_endpoints_dict_has_keys(self):
        assert isinstance(N8N_ENDPOINTS, dict)
        assert len(N8N_ENDPOINTS) > 0

    def test_endpoints_has_monitoring(self):
        assert "01_monitoring" in N8N_ENDPOINTS

    def test_endpoints_has_profiles(self):
        assert "02_profiles" in N8N_ENDPOINTS

    def test_endpoints_has_market(self):
        assert "03_market" in N8N_ENDPOINTS

    def test_each_endpoint_has_method_and_path(self):
        for group, endpoints in N8N_ENDPOINTS.items():
            for ep in endpoints:
                assert "M" in ep
                assert "P" in ep

    def test_each_endpoint_has_rate_limit(self):
        for group, endpoints in N8N_ENDPOINTS.items():
            for ep in endpoints:
                assert "RL" in ep


class TestN8NWorkflows:
    def test_workflows_dict(self):
        assert isinstance(N8N_WORKFLOWS, dict)
        assert len(N8N_WORKFLOWS) > 0

    def test_weekly_report_exists(self):
        assert "weekly_report" in N8N_WORKFLOWS

    def test_nightly_pipeline_exists(self):
        assert "nightly_pipeline" in N8N_WORKFLOWS

    def test_trend_alert_exists(self):
        assert "trend_alert" in N8N_WORKFLOWS


class TestN8NWebhooks:
    def test_webhooks_dict(self):
        assert isinstance(N8N_WEBHOOKS, dict)

    def test_student_created_webhook(self):
        assert "student-created" in N8N_WEBHOOKS
        wh = N8N_WEBHOOKS["student-created"]
        assert wh["method"] == "POST"
        assert "body" in wh

    def test_alert_webhook(self):
        assert "alert" in N8N_WEBHOOKS
        wh = N8N_WEBHOOKS["alert"]
        assert wh["method"] == "POST"
        assert "body" in wh

    def test_webhooks_have_required_fields(self):
        for name, wh in N8N_WEBHOOKS.items():
            assert "method" in wh
            assert "path" in wh
            assert "description" in wh
