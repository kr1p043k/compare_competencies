import asyncio
from unittest.mock import patch

from src.decorators import timed, timed_block, timed_async


class TestTimed:
    def test_timed_sync(self):
        @timed("test_fn")
        def foo():
            return 42

        with patch("src.decorators.logger.info") as mock_log:
            result = foo()
        assert result == 42
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        assert args[0] == "timed"
        assert "function" in mock_log.call_args.kwargs
        assert mock_log.call_args.kwargs["function"] == "test_fn"
        assert "elapsed_sec" in mock_log.call_args.kwargs
        assert isinstance(mock_log.call_args.kwargs["elapsed_sec"], float)

    def test_timed_auto_name(self):
        @timed()
        def bar():
            return 1

        with patch("src.decorators.logger.info") as mock_log:
            bar()
        assert mock_log.call_args.kwargs["function"] == "bar"

    def test_timed_async(self):
        @timed()
        async def async_fn():
            return 99

        with patch("src.decorators.logger.info") as mock_log:
            result = asyncio.run(async_fn())
        assert result == 99
        mock_log.assert_called_once()

    def test_timed_async_alias(self):
        deco = timed_async("alias_test")
        assert callable(deco)


class TestTimedBlock:
    def test_timed_block(self):
        with patch("src.decorators.structlog.get_logger") as mock_get_log:
            mock_logger = mock_get_log.return_value
            with timed_block("test_block") as tb:
                tb.log_key = "test_block"
        mock_logger.info.assert_called_once()
        assert mock_logger.info.call_args.kwargs["function"] == "test_block"

    def test_timed_block_default_key(self):
        with patch("src.decorators.structlog.get_logger"):
            with timed_block("custom_key") as tb:
                assert tb.log_key == "custom_key"
