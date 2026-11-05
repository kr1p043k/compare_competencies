"""WS Origin-check: защита от cross-site WebSocket hijacking.

Без проверки Origin сторонний сайт открывает сокет от имени залогиненного
пользователя: браузер прикладывает cookie к апгрейду, и SameSite=Lax не
спасает при навигации верхнего уровня.
"""

from src.api_pkg.ws_origin import origin_allowed


class _WS:
    """Минимальный WebSocket: нужны только headers и url.path."""

    def __init__(self, origin: str = "", host: str = "localhost:8000", path: str = "/api/pipeline/ws"):
        self.headers = {"host": host}
        if origin:
            self.headers["origin"] = origin
        self.url = type("U", (), {"path": path})()


ALLOWED = "http://localhost:3000,https://cc.example.ru"


class TestOriginAllowed:
    def test_no_origin_is_allowed(self):
        """Не-браузерные клиенты (скрипты, тесты) Origin не шлют."""
        assert origin_allowed(_WS(), ALLOWED) is True

    def test_same_host_as_socket_allowed(self):
        assert origin_allowed(_WS("http://localhost:8000"), ALLOWED) is True

    def test_configured_origin_allowed(self):
        assert origin_allowed(_WS("http://localhost:3000"), ALLOWED) is True
        assert origin_allowed(_WS("https://cc.example.ru"), ALLOWED) is True

    def test_foreign_origin_rejected(self):
        assert origin_allowed(_WS("https://evil.example.com"), ALLOWED) is False

    def test_prefix_trick_rejected(self):
        """cc.example.ru.evil.com не должен проходить как cc.example.ru."""
        assert origin_allowed(_WS("https://cc.example.ru.evil.com"), ALLOWED) is False

    def test_port_matters(self):
        assert origin_allowed(_WS("http://localhost:3001"), ALLOWED) is False

    def test_wildcard_allows_any(self):
        assert origin_allowed(_WS("https://anything.example"), "*") is True

    def test_broken_origin_rejected(self):
        """Origin без схемы/хоста не даёт совпадения."""
        assert origin_allowed(_WS("null"), ALLOWED) is False
