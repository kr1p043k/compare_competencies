"""Minimal local replacement for ``aioresponses`` used by the test-suite.

Why this exists
---------------
``aioresponses`` reaches into private ``aiohttp`` internals and constructs a
real ``aiohttp.ClientResponse``. That constructor gained a required
``stream_writer`` argument in newer aiohttp releases, so ``aioresponses``
breaks as soon as ``aiohttp`` is upgraded (``ClientResponse.__init__() missing
1 required positional argument: 'stream_writer'``).

This shim patches ``aiohttp.ClientSession._request`` with a plain async
function and returns a lightweight fake response object that implements only
the surface the tests actually use (``status``, ``headers``, ``json()``,
``text()``, ``release()`` and the async context-manager protocol). It has no
dependency on aiohttp private APIs, so it keeps working across aiohttp
versions and lets us bump aiohttp for the security fixes.

Supported API (the subset used by ``tests/parsing/test_api.py``)
----------------------------------------------------------------
* context manager: ``with aioresponses() as m: ...``
* ``m.get/post/put/patch/delete/head/options(url, **kwargs)``
* ``m.add(url, method="GET", **kwargs)``
* kwargs: ``payload``, ``body``, ``status``, ``headers``, ``exception``,
  ``repeat`` (plus ``content_type``, ``reason`` tolerated for parity)
* ``url`` may be a plain string or a compiled ``re.Pattern``
* repeated registrations for the same URL are returned in registration order
  and consumed one by one (``repeat=False`` by default, mirroring aioresponses)
* ``m.requests`` tracks every request as ``{(METHOD, url): [RequestCall, ...]}``
* unmatched requests raise a descriptive ``AssertionError``
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp

__all__ = ["aioresponses", "MockResponse", "RequestCall"]

_REGEX_TYPES = (re.Pattern,)


def _is_pattern(url: Any) -> bool:
    return isinstance(url, _REGEX_TYPES)


def _merge_params(url: str, params: Any) -> str:
    """Return ``url`` with ``params`` encoded into the query string.

    Mirrors what ``aioresponses`` does before matching, so registrations
    written without query parameters still match requests that pass ``params``.
    """
    if not params:
        return str(url)
    parts = urlsplit(str(url))
    extra = urlencode(params, doseq=True)
    query = f"{parts.query}&{extra}" if parts.query else extra
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


class RequestCall:
    """A single intercepted request (its method, url and original kwargs)."""

    __slots__ = ("method", "url", "kwargs")

    def __init__(self, method: str, url: str, kwargs: dict[str, Any]):
        self.method = method
        self.url = url
        self.kwargs = kwargs

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"RequestCall({self.method!r}, {self.url!r})"


class _CaseInsensitiveHeaders(dict):
    """Tiny case-insensitive header mapping (like aiohttp's CIMultiDict)."""

    def __init__(self, headers: dict[str, str] | None = None):
        super().__init__()
        if headers:
            for key, value in headers.items():
                self[key] = value

    def _find_key(self, key: str) -> str | None:
        lowered = key.lower()
        for existing in self.keys():
            if existing.lower() == lowered:
                return existing
        return None

    def __setitem__(self, key: str, value: str) -> None:
        existing = self._find_key(key)
        super().__setitem__(existing if existing is not None else key, value)

    def __getitem__(self, key: str) -> str:
        existing = self._find_key(key)
        if existing is None:
            raise KeyError(key)
        return super().__getitem__(existing)

    def get(self, key: str, default: Any = None) -> Any:
        existing = self._find_key(key)
        return super().__getitem__(existing) if existing is not None else default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._find_key(key) is not None


class MockResponse:
    """Lightweight stand-in for ``aiohttp.ClientResponse``.

    Only the attributes and coroutines used by the production parser and the
    tests are implemented.
    """

    def __init__(
        self,
        *,
        method: str = "GET",
        url: str = "",
        status: int = 200,
        body: Any = b"",
        payload: Any = None,
        headers: dict[str, str] | None = None,
        content_type: str = "application/json",
        reason: str | None = None,
    ):
        if payload is not None:
            body = json.dumps(payload)
        if body is None:
            body = b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        elif not isinstance(body, (bytes, bytearray)):
            body = str(body).encode("utf-8")

        merged_headers = _CaseInsensitiveHeaders(headers)
        merged_headers.setdefault("Content-Type", content_type)
        self._body: bytes = bytes(body)
        self.method = method
        self.url = url
        self.status = status
        self.reason = reason or ""
        self.headers = merged_headers
        self.content_type = content_type
        self.closed = False
        self.released = False
        self._history: tuple = ()

    @property
    def history(self) -> tuple:
        return self._history

    async def json(self, **_kwargs: Any) -> Any:
        return json.loads(self._body.decode("utf-8"))

    async def text(self, encoding: str | None = None) -> str:
        return self._body.decode(encoding or "utf-8", errors="replace")

    async def read(self) -> bytes:
        return self._body

    def release(self) -> None:
        self.released = True

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None,  # type: ignore[arg-type]
                history=(),
                status=self.status,
                message=self.reason or f"HTTP {self.status}",
            )

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"MockResponse(status={self.status}, url={self.url!r})"


class _Match:
    """One registered (method, url) -> response rule."""

    def __init__(
        self,
        url: Any,
        *,
        method: str = "GET",
        status: int = 200,
        body: Any = b"",
        payload: Any = None,
        exception: BaseException | None = None,
        headers: dict[str, str] | None = None,
        content_type: str = "application/json",
        repeat: Any = False,
        reason: str | None = None,
        **_ignored: Any,
    ):
        self.url = url
        self.is_pattern = _is_pattern(url)
        self.method = method.upper()
        self.status = status
        self.body = body
        self.payload = payload
        self.exception = exception
        self.headers = headers
        self.content_type = content_type
        self.repeat = repeat
        self.reason = reason

    def matches(self, method: str, url: str) -> bool:
        if self.method != method.upper():
            return False
        if self.is_pattern:
            return bool(self.url.match(str(url)))
        return str(self.url) == str(url)

    def build(self, method: str, url: str) -> MockResponse:
        return MockResponse(
            method=method,
            url=url,
            status=self.status,
            body=self.body,
            payload=self.payload,
            headers=self.headers,
            content_type=self.content_type,
            reason=self.reason,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"_Match({self.method} {self.url!r})"


class aioresponses:  # noqa: N801 - name mirrors the aioresponses API
    """Drop-in mock for the subset of ``aioresponses.aioresponses`` we use."""

    def __init__(self, **_kwargs: Any):
        self._matches: list[_Match] = []
        self.requests: dict[tuple[str, str], list[RequestCall]] = {}
        self._original_request = None
        self._started = False

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> aioresponses:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        self._original_request = aiohttp.ClientSession._request
        mock = self

        async def _patched_request(
            session_self: aiohttp.ClientSession,
            method: str,
            url: Any,
            **kwargs: Any,
        ) -> MockResponse:
            return await mock._handle(session_self, method, url, **kwargs)

        aiohttp.ClientSession._request = _patched_request  # type: ignore[assignment]
        self._started = True

    def stop(self) -> None:
        if self._original_request is not None:
            aiohttp.ClientSession._request = self._original_request  # type: ignore[assignment]
            self._original_request = None
        self._started = False

    # -- registration ------------------------------------------------------
    def head(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="HEAD", **kwargs)

    def get(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="GET", **kwargs)

    def post(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="POST", **kwargs)

    def put(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="PUT", **kwargs)

    def patch(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="PATCH", **kwargs)

    def delete(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="DELETE", **kwargs)

    def options(self, url: Any, **kwargs: Any) -> None:
        self.add(url, method="OPTIONS", **kwargs)

    def add(self, url: Any, method: str = "GET", **kwargs: Any) -> None:
        self._matches.append(_Match(url, method=method, **kwargs))

    def clear(self) -> None:
        self._matches.clear()

    # -- request interception ---------------------------------------------
    async def _handle(
        self,
        _session: aiohttp.ClientSession,
        method: str,
        url: Any,
        **kwargs: Any,
    ) -> MockResponse:
        full_url = _merge_params(str(url), kwargs.get("params"))
        method_upper = method.upper()
        self.requests.setdefault((method_upper, full_url), []).append(
            RequestCall(method_upper, full_url, dict(kwargs))
        )

        match = self._find(method_upper, full_url)
        if match is None:
            raise AssertionError(
                f"No mocked response for {method_upper} {full_url}.\n"
                f"Registered: {self._registered_summary()}"
            )

        if match.exception is not None:
            raise match.exception

        return match.build(method_upper, full_url)

    def _find(self, method: str, url: str) -> _Match | None:
        for index, match in enumerate(self._matches):
            if not match.matches(method, url):
                continue

            # Mirror aioresponses consumption semantics: bool False -> single
            # use; bool True -> repeat forever; int N -> N uses.
            if isinstance(match.repeat, bool):
                if not match.repeat:
                    del self._matches[index]
            else:
                if match.repeat <= 1:
                    del self._matches[index]
                else:
                    match.repeat -= 1
            return match
        return None

    def _registered_summary(self) -> str:
        if not self._matches:
            return "<nothing registered>"
        return ", ".join(repr(m) for m in self._matches)

    # -- minimal assertion helpers (parity with aioresponses) -------------
    def assert_called(self) -> None:
        if not any(self.requests.values()):
            raise AssertionError("Expected aioresponses to have been called.")

    def assert_not_called(self) -> None:
        if any(self.requests.values()):
            raise AssertionError("Expected aioresponses to not have been called.")
