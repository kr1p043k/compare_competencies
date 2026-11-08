"""Export the FastAPI OpenAPI schema to a deterministic openapi.json.

Purpose: keep the committed openapi.json in sync with the backend so the
frontend can generate TypeScript types from it (openapi-typescript) without
the two drifting apart. CI re-runs this exporter into a temp file and fails
if the committed artifact differs (see .github/workflows/ci.yml).

Usage:
    python scripts/export_openapi.py [--output PATH]

Requirements: works offline, no database needed. Only SECRET_KEY is required
to import the app factory, because pydantic-settings validates it the moment
`src.config.Settings` is instantiated (src/config.py:300-322).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "openapi.json"

# Running `python scripts/export_openapi.py` puts `scripts/` (not the repo
# root) first on sys.path, so `src` is not importable without this. It does
# not trigger any app import itself, so the SECRET_KEY check below still runs
# before the heavy `src.api_pkg` import in _write_schema.
sys.path.insert(0, str(ROOT))


def _secret_key_from_environment() -> str | None:
    """Return SECRET_KEY from the environment or the repo .env, else None.

    pydantic-settings reads `.env` itself on import, but we need to know here
    (before the import) whether a key exists so a missing key produces the
    clear message in `main` instead of a raw pydantic ValidationError.
    """
    key = os.getenv("SECRET_KEY")
    if key:
        return key

    env_file = ROOT / ".env"
    if not env_file.is_file():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != "SECRET_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def _write_schema(output: Path) -> dict[str, Any]:
    """Generate the app's OpenAPI schema and write it to `output` as JSON."""
    # Imported only after the SECRET_KEY check so an absent key fails with the
    # message in `main`, not with a pydantic ValidationError traceback.
    from src.api_pkg import create_app

    schema: dict[str, Any] = create_app().openapi()

    output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the schema (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    if not _secret_key_from_environment():
        print(
            "SECRET_KEY is not set. Importing the app requires it because\n"
            "src/config.py validates SECRET_KEY when Settings is created.\n\n"
            "Set it in the environment or in a local .env with a strong value:\n"
            "    SECRET_KEY=$(openssl rand -hex 32)\n\n"
            "Then re-run: python scripts/export_openapi.py",
            file=sys.stderr,
        )
        return 1

    schema = _write_schema(args.output)

    n_paths = len(schema.get("paths", {}))
    n_schemas = len(schema.get("components", {}).get("schemas", {}))
    print(
        f"Wrote {args.output} "
        f"(size={args.output.stat().st_size} bytes, "
        f"paths={n_paths}, schemas={n_schemas})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
