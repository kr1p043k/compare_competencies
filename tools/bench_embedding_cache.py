"""Benchmark the skill-embedding cache hydration cost of ``EmbeddingComparator``.

Measures, in isolation, the two numbers the npz cache format targets:

  1. cold load   - ``EmbeddingComparator._load_skill_cache()`` on a fresh instance;
  2. warm analyze - ``embed_skills(all_cached_skills)`` once the cache is hydrated.

The sentence-transformer provider is stubbed: loading model weights (~10s) is
orthogonal to cache hydration (identical before/after this change) and would only
drown the signal. Only the on-disk cache drives these timings.

If ``skill_embeddings.json`` exists but ``skill_embeddings.npz`` does not, the
first cold load also performs the one-time JSON -> npz migration; the file kind
(migrate / npz / json) is printed per repeat so the transition is visible.

Usage:
    python tools/bench_embedding_cache.py [--cache-dir data/cache/embeddings] [--repeats 5]
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The worktree has no .env; config validates SECRET_KEY at import time. Generate a
# throwaway key so the benchmark runs standalone (never used for real signing).
if not os.environ.get("SECRET_KEY"):
    os.environ["SECRET_KEY"] = secrets.token_hex(32)

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.analyzers.comparison import embedding_provider as ep  # noqa: E402

_DIM = 768


class _StubProvider(ep.EmbeddingProvider):
    """No-model stand-in: cache hydration never calls encode for a warm set."""

    @property
    def dimension(self) -> int:
        return _DIM

    def model_version(self) -> str:
        return "bench-stub"

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        raise AssertionError("stub encode must not be called in this benchmark")


_STUB = _StubProvider()
# Pre-seed the singleton slot so EmbeddingComparator.__init__ never loads weights.
ep.EmbeddingProviderFactory._instances["default"] = _STUB


def _fresh(cache_dir: Path):
    from src.analyzers.comparison.embedding_comparator import EmbeddingComparator

    return EmbeddingComparator(cache_dir=str(cache_dir))


def measure(cache_dir: Path, repeats: int) -> tuple[list[float], list[float]]:
    cold: list[float] = []
    warm: list[float] = []
    for _ in range(repeats):
        comp = _fresh(cache_dir)
        t0 = time.perf_counter()
        comp._load_skill_cache()
        cold.append(time.perf_counter() - t0)

        keys = list(comp._skill_cache.keys())
        t1 = time.perf_counter()
        comp.embed_skills(keys)
        warm.append(time.perf_counter() - t1)
    return cold, warm


def _num_skills(cache_dir: Path) -> int:
    comp = _fresh(cache_dir)
    comp._load_skill_cache()
    return len(comp._skill_cache)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else config.EMBEDDINGS_CACHE_DIR
    json_path = cache_dir / "skill_embeddings.json"
    npz_path = cache_dir / "skill_embeddings.npz"
    json_size = json_path.stat().st_size if json_path.exists() else 0
    npz_size = npz_path.stat().st_size if npz_path.exists() else 0
    npz_existed_before = npz_path.exists()
    print(f"cache dir: {cache_dir}")
    print(f"json present: {json_path.exists()} ({json_size} B)")
    print(f"npz  present: {npz_path.exists()} ({npz_size} B)")

    cold, warm = measure(cache_dir, args.repeats)
    num_skills = _num_skills(cache_dir)

    def kind(i: int) -> str:
        if npz_existed_before:
            return "npz"
        if json_path.exists() and npz_path.exists():
            return "migrate" if i == 0 else "npz"
        return "json"

    print(f"\n{'repeat':>6}  {'cache-file':<12} {'cold load (s)':>14} {'warm analyze (s)':>16}")
    for i, (c, w) in enumerate(zip(cold, warm, strict=False)):
        print(f"{i:>6}  {kind(i):<12} {c:>14.4f} {w:>16.4f}")

    print(f"\nskills in cache : {num_skills}")
    print(f"cold load  best : {min(cold):.4f} s")
    print(f"warm analyze best: {min(warm):.4f} s")
    final_npz = npz_path.stat().st_size if npz_path.exists() else None
    print(f"npz size after  : {final_npz if final_npz is not None else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
