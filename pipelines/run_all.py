"""Rebuild everything from scratch: raw data, processed tables, models and their exports.

    python pipelines/run_all.py                    # ingestion + training (~9 min)
    python pipelines/run_all.py --tune             # also re-run the Optuna search (+~10 min)
    python pipelines/run_all.py --experiments      # also the 7.3 / 7.4 / 7.5 comparisons (+~10 min)

Everything is seeded (config.yaml: project.seed), so a clean checkout produces the same data, the
same champions and the same metrics. Only dates, versions and the git commit in the *_meta.json files
change. Any extra arguments are passed on to run_training.py.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines import run_ingestion, run_training  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    began = time.perf_counter()
    print("=== Stage 1: ingestion ===")
    if run_ingestion.main() != 0:
        return 1
    print("\n=== Stage 2: training ===")
    if run_training.main(argv) != 0:
        return 1
    print(f"\nrun_all finished in {time.perf_counter() - began:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
