"""Тесты EventBus."""
from src.pipeline.event_bus import EventBus, PipelineEvent


class TestEventBus:
    def test_emit_and_receive(self):
        bus = EventBus()
        received = []

        def handler(event: PipelineEvent):
            received.append(event)

        bus.on("ok", handler)
        bus.emit(PipelineEvent(stage="test", status="ok", elapsed=1.0))

        assert len(received) == 1
        assert received[0].stage == "test"
        assert received[0].status == "ok"

    def test_wildcard_handler(self):
        bus = EventBus()
        received = []

        def handler(event: PipelineEvent):
            received.append(event)

        bus.on("*", handler)
        bus.emit(PipelineEvent(stage="a", status="ok", elapsed=0.1))
        bus.emit(PipelineEvent(stage="b", status="fail", elapsed=0.2))

        assert len(received) == 2

    def test_off_handler(self):
        bus = EventBus()
        received = []

        def handler(event: PipelineEvent):
            received.append(event)

        bus.on("ok", handler)
        bus.off("ok", handler)
        bus.emit(PipelineEvent(stage="test", status="ok", elapsed=1.0))

        assert len(received) == 0

    def test_history(self):
        bus = EventBus()
        bus.emit(PipelineEvent(stage="a", status="ok", elapsed=0.1))
        bus.emit(PipelineEvent(stage="b", status="fail", elapsed=0.2))

        assert len(bus.history) == 2
        assert bus.history[0].stage == "a"

    def test_clear(self):
        bus = EventBus()
        bus.emit(PipelineEvent(stage="a", status="ok", elapsed=0.1))
        bus.clear()
        assert len(bus.history) == 0

    def test_handler_exception_does_not_crash(self):
        bus = EventBus()

        def failing(event: PipelineEvent):
            raise ValueError("boom")

        bus.on("ok", failing)
        bus.emit(PipelineEvent(stage="test", status="ok", elapsed=1.0))
        assert len(bus.history) == 1
