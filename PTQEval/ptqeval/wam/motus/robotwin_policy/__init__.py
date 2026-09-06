"""RoboTwin policy package for the motus ViDiT-Q variant.

Symlinked into RoboTwin/policy/<name> by the eval runner; RoboTwin's
script/eval_policy.py imports it and reads get_model / eval / reset_model.
Delegates to ptqeval.wam.motus.policy (variant dispatch + on-policy calib).
"""
from ptqeval.wam.motus.policy import eval, get_model, reset_model  # noqa: F401

__all__ = ["get_model", "eval", "reset_model"]
