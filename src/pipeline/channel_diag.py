"""Explicit degradation signal for pipeline channels that swallow failures.

A channel that fails silently must not masquerade as an ``ok`` result:
``ChannelDiag`` accumulates failed channel names and turns them into a
machine-readable ``diag`` block that survives into the final result, so a
degraded run is visible and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Public API shape of ``as_dict()``.
DiagDict = dict[str, object]


@dataclass
class ChannelDiag:
    """Tracks failed channels and produces an API-safe ``diag`` dict.

    Usage::

        diag = ChannelDiag()
        try:
            build_embedding_index(...)
        except Exception:
            diag.fail("gap_enhance.embeddings")
        summary["diag"] = diag.as_dict()
    """

    _failed: list[str] = field(default_factory=list, init=False)

    def fail(self, name: str) -> None:
        """Record a failed channel. Duplicates are ignored."""
        if name and name not in self._failed:
            self._failed.append(name)

    @property
    def failed_channels(self) -> list[str]:
        """Sorted list of unique failed channel names."""
        return sorted(self._failed)

    @property
    def degraded(self) -> bool:
        """True if at least one channel failed."""
        return bool(self._failed)

    def as_dict(self) -> DiagDict:
        """API-safe dict: degraded flag + failed channel names + count."""
        channels = self.failed_channels
        return {
            "degraded": bool(channels),
            "failed_channels": channels,
            "count": len(channels),
        }
