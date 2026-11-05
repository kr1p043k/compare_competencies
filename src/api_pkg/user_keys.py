"""Shared derivation of a user's private `self_<username>` profile key.

The `self_<username>` profile is created and persisted on disk under the name
returned by :func:`self_profile_name` (see ``profiles._self_profile_name``).
Every guard that checks "is this the user's own profile?" must build the key
with the same function, or a non-ASCII username (e.g. Cyrillic) resolves to a
different key and the user is locked out of their own profile (403).
"""

import re

_SELF_MAX_LEN = 40
_SELF_FALLBACK = "anon"


def self_profile_name(username: str) -> str:
    """Canonical name of the user's private profile: ``self_<ascii-safe name>``.

    Mirrors exactly the derivation used at profile creation time: lowercase,
    keep only ``[a-z0-9_]``, truncate to 40 chars, fall back to ``anon``.
    """
    safe = re.sub(r"[^a-z0-9_]", "_", (username or "anon").strip().lower())[:_SELF_MAX_LEN]
    return f"self_{safe or _SELF_FALLBACK}"


def self_profile_keys(username: str) -> set[str]:
    """All profile keys that resolve to the user's own profile (single key)."""
    return {self_profile_name(username)}
