from src.api_pkg.n8n import N8N_ENDPOINTS, N8N_WORKFLOWS, N8N_WEBHOOKS


def test_n8n_endpoints():
    assert len(N8N_ENDPOINTS) == 10
    total = sum(len(v) for v in N8N_ENDPOINTS.values())
    assert total == 51


def test_n8n_workflows():
    assert len(N8N_WORKFLOWS) == 6


def test_n8n_webhooks():
    assert len(N8N_WEBHOOKS) == 3
    assert "student-created" in N8N_WEBHOOKS
