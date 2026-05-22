"""Result[T, E] / Either pattern для явной обработки ошибок без исключений."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Result(Generic[T, E]):
    """Базовый класс для Ok/Err. Используйте Ok(value) или Err(error)."""

    def is_ok(self) -> bool:
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        return isinstance(self, Err)

    def unwrap(self) -> T:
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T:
        raise NotImplementedError

    def ok(self) -> T | None:
        if isinstance(self, Ok):
            return self._value  # type: ignore[no-any-return]
        return None

    def err(self) -> E | None:
        if isinstance(self, Err):
            return self._error  # type: ignore[no-any-return]
        return None


class Ok(Result[T, E]):
    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


class Err(Result[T, E]):
    __slots__ = ("_error",)

    def __init__(self, error: E) -> None:
        self._error = error

    def unwrap(self) -> T:
        raise RuntimeError(f"Called unwrap() on Err: {self._error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def __repr__(self) -> str:
        return f"Err({self._error!r})"
