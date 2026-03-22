import _bootstrap  # noqa: F401

from src.pipeline.simulate import run_sanity_checks
from src.utils.logger import save_json


if __name__ == "__main__":
    checks = run_sanity_checks(seed=2026)
    save_json("results/raw/ber_logs/exp10_sanity_checks.json", checks)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("Sanity checks failed:")
        for key in failed:
            print(f" - {key}")
    else:
        print("All sanity checks passed.")
