# Package init for the fastwam WAM adapter.
#
# Sibling repo `FastWAM/` provides the `fastwam` package (pip-installed into
# FastWAM/.venv via `uv pip install -e FastWAM/`) plus hydra configs under
# FastWAM/configs and the deploy policy under FastWAM/experiments/robotwin.
# We expose those source paths so config composition and the RoboTwin policy
# import resolve from anywhere inside this package.
import os
import sys

FASTWAM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "FastWAM")
)
FASTWAM_SRC = os.path.join(FASTWAM_ROOT, "src")
FASTWAM_POLICY_DIR = os.path.join(FASTWAM_ROOT, "experiments", "robotwin")

for _p in (FASTWAM_ROOT, FASTWAM_SRC, FASTWAM_POLICY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
