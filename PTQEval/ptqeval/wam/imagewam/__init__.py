# Package init for the imagewam WAM adapter.
#
# Sibling (vendored) repo `ImageWAM/` provides the `imagewam` package
# (src/imagewam, pip-installed via `uv pip install -e ImageWAM/`), the FLUX.2
# source it depends on (third_party/flux2/src -> `flux2` package), the RoboTwin
# deploy policy (experiments/robotwin/imagewam_policy), the hydra configs
# (ImageWAM/configs), and the vendored RoboTwin sim (third_party/RoboTwin).
# We expose those source paths so config composition and the RoboTwin policy
# import resolve from anywhere inside this package.
import os
import sys

IMAGEWAM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ImageWAM")
)
IMAGEWAM_SRC = os.path.join(IMAGEWAM_ROOT, "src")
IMAGEWAM_FLUX2_SRC = os.path.join(IMAGEWAM_ROOT, "third_party", "flux2", "src")
IMAGEWAM_POLICY_DIR = os.path.join(IMAGEWAM_ROOT, "experiments", "robotwin")

for _p in (IMAGEWAM_ROOT, IMAGEWAM_SRC, IMAGEWAM_FLUX2_SRC, IMAGEWAM_POLICY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
