"""Full rebuild: clean cache → pipeline → clusters → model → gap analysis.

Usage:
    python -m src.cli rebuild
"""

import sys

from src.logging_config import setup_structlog
from src.pipeline import runner as pr

setup_structlog()


def main() -> None:
    print("=" * 60)
    print("   ПОЛНАЯ ПЕРЕСБОРКА")
    print("=" * 60)
    result = pr.rebuild()
    if result.is_err():
        print(f"\n>>> ОШИБКА ПЕРЕСБОРКИ: {result.err()}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print("   ПЕРЕСБОРКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
