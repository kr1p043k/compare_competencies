"""Lightweight Dependency Injection container.

Provides a simple registry for services with lazy initialization.
Uses punq if available, falls back to a simple dict-based container.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import structlog

T = TypeVar("T")
logger = structlog.get_logger("di")


class DIContainer:
    def __init__(self):
        self._registry: dict[str, dict] = {}
        self._instances: dict[str, Any] = {}
        self._parent: DIContainer | None = None

    def register(self, key: type | str, factory: Callable[..., Any] | None = None, instance: Any = None) -> None:
        name = key.__name__ if isinstance(key, type) else key
        if instance is not None:
            self._instances[name] = instance
            self._registry.pop(name, None)
            logger.debug("di_registered_instance", name=name)
            return
        self._registry[name] = {"factory": factory, "singleton": True}
        self._instances.pop(name, None)
        logger.debug("di_registered_factory", name=name)

    def register_transient(self, key: type | str, factory: Callable[..., Any]) -> None:
        name = key.__name__ if isinstance(key, type) else key
        self._registry[name] = {"factory": factory, "singleton": False}
        self._instances.pop(name, None)

    def resolve(self, key: type[T] | str) -> T:
        name = key.__name__ if isinstance(key, type) else key
        if name in self._instances:
            return self._instances[name]
        if name in self._registry:
            entry = self._registry[name]
            instance = entry["factory"]()
            if entry["singleton"]:
                self._instances[name] = instance
            return instance
        if self._parent:
            return self._parent.resolve(key)
        raise KeyError(f"Service not registered: {name}")

    def has(self, key: type | str) -> bool:
        name = key.__name__ if isinstance(key, type) else key
        return name in self._instances or name in self._registry or (self._parent and self._parent.has(key))

    def clear(self) -> None:
        self._instances.clear()
        self._registry.clear()

    def create_child(self) -> DIContainer:
        child = DIContainer()
        child._parent = self
        return child

    def list_services(self) -> list[str]:
        keys = set(self._registry.keys()) | set(self._instances.keys())
        return sorted(keys)


_container: DIContainer | None = None


def get_container() -> DIContainer:
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container() -> None:
    global _container
    _container = None
