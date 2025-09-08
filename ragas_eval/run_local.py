"""
Convenience runner to execute ragas_eval.main from inside the ragas_eval folder.
Usage:
  python run_local.py --dry_run --max_rows 2
"""
from pathlib import Path
import sys


def _ensure_package_on_path() -> None:
    # Add repository root (parent of this folder) to sys.path so that
    # `import ragas_eval` works even when running from inside ragas_eval/.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))


def main() -> None:
    _ensure_package_on_path()
    from ragas_eval.main import main as pkg_main

    pkg_main()


if __name__ == "__main__":
    main()

