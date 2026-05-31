import pytest

from src.di import DIContainer, get_container, reset_container


class TestDIContainer:
    def test_register_instance(self):
        c = DIContainer()
        c.register("db", instance="pg")
        assert c.resolve("db") == "pg"

    def test_register_factory_singleton(self):
        c = DIContainer()
        c.register("cfg", factory=lambda: {"key": "val"})
        r1 = c.resolve("cfg")
        r2 = c.resolve("cfg")
        assert r1 is r2
        assert r1["key"] == "val"

    def test_register_transient(self):
        c = DIContainer()
        c.register_transient("svc", factory=lambda: object())
        r1 = c.resolve("svc")
        r2 = c.resolve("svc")
        assert r1 is not r2

    def test_register_by_type(self):
        c = DIContainer()
        c.register(int, factory=lambda: 42)
        assert c.resolve(int) == 42

    def test_has_service(self):
        c = DIContainer()
        assert not c.has("missing")
        c.register("svc", instance=1)
        assert c.has("svc")

    def test_clear(self):
        c = DIContainer()
        c.register("svc", instance=1)
        c.clear()
        assert not c.has("svc")
        with pytest.raises(KeyError):
            c.resolve("svc")

    def test_resolve_unregistered(self):
        c = DIContainer()
        with pytest.raises(KeyError):
            c.resolve("nonexistent")

    def test_create_child(self):
        parent = DIContainer()
        parent.register("parent_svc", instance="p")
        child = parent.create_child()
        assert child.resolve("parent_svc") == "p"
        child.register("child_svc", instance="c")
        assert child.has("child_svc")
        assert not parent.has("child_svc")

    def test_list_services(self):
        c = DIContainer()
        c.register("a", instance=1)
        c.register("b", instance=2)
        svcs = c.list_services()
        assert "a" in svcs
        assert "b" in svcs

    def test_instance_replaces_factory(self):
        c = DIContainer()
        c.register("svc", factory=lambda: "factory_val")
        c.register("svc", instance="instance_val")
        assert c.resolve("svc") == "instance_val"

    def test_factory_replaces_instance(self):
        c = DIContainer()
        c.register("svc", instance="old")
        c.register("svc", factory=lambda: "new")
        c._registry["svc"]["singleton"] = False
        assert c.resolve("svc") == "new"

    def test_resolve_overrides_instance(self):
        c = DIContainer()
        c.register("svc", instance="old")
        assert c.resolve("svc") == "old"

    def test_child_overrides_parent(self):
        parent = DIContainer()
        parent.register("svc", instance="parent_val")
        child = parent.create_child()
        child.register("svc", instance="child_val")
        assert child.resolve("svc") == "child_val"

    def test_child_resolve_from_parent(self):
        parent = DIContainer()
        parent.register("parent_only", instance="yes")
        child = parent.create_child()
        assert child.resolve("parent_only") == "yes"

    def test_container_has_type_key(self):
        c = DIContainer()
        c.register(str, instance="hello")
        assert c.has(str)
        assert c.resolve(str) == "hello"


class TestGlobalContainer:
    def setup_method(self):
        reset_container()

    def test_get_container_singleton(self):
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2

    def test_reset_container(self):
        c1 = get_container()
        c1.register("svc", instance=1)
        reset_container()
        c2 = get_container()
        assert c2 is not c1
        assert not c2.has("svc")
