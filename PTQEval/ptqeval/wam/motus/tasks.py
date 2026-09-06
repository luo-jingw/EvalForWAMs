"""RoboTwin task lists for the motus WAM.

Re-exported from the lingbot_va list (shared RoboTwin 2.0 benchmark). NOTE:
verify Motus's vendored RoboTwin / data/robotwin2 task set matches when the env
is up; if it diverges, read Motus's own task list instead.
"""
from __future__ import annotations

from ptqeval.wam.lingbot_va.tasks import (  # noqa: F401
    ALL_TASKS,
    EVAL_STEP_LIMIT,
    SELECTED_15_TASKS,
    SMOKE_5_TASKS,
)

__all__ = ["ALL_TASKS", "SELECTED_15_TASKS", "SMOKE_5_TASKS", "EVAL_STEP_LIMIT"]
