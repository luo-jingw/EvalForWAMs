"""RoboTwin policy package for the fastwam ViDiT-Q variant.

Symlinked into RoboTwin/policy/<name> by the eval runner; RoboTwin's
script/eval_policy.py does `importlib.import_module(<name>)` and reads
get_model / eval / reset_model off it. Delegates to ptqeval.wam.fastwam.policy
(variant dispatch + PerfProbe).
"""
from ptqeval.wam.fastwam.policy import eval, get_model, reset_model  # noqa: F401

__all__ = ["get_model", "eval", "reset_model"]
