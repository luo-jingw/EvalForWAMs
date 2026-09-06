# Package init for the motus WAM adapter.
#
# Vendored repo `Motus/` (thu-ml/Motus) provides the `models`/`utils` packages,
# the WAN modules under `bak/` (the model does `from wan.modules... import ...`),
# and the RoboTwin deploy policy under `inference/robotwin/Motus`. Expose those
# source paths so model construction and the RoboTwin policy import resolve from
# anywhere inside this package.
import os
import sys

MOTUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Motus")
)
MOTUS_BAK = os.path.join(MOTUS_ROOT, "bak")  # `wan` package lives here
MOTUS_POLICY_DIR = os.path.join(MOTUS_ROOT, "inference", "robotwin", "Motus")

for _p in (MOTUS_ROOT, MOTUS_BAK, MOTUS_POLICY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
