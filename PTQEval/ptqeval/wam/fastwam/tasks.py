"""RoboTwin task lists for the fastwam WAM.

The RoboTwin task set is benchmark data shared with lingbot_va (plan.txt
§1.1.1: the two WAMs' vendored RoboTwin envs are byte-identical, so the same
task names + eval step limits apply). Re-exported from the lingbot_va list to
keep a single source of truth. Importing lingbot_va.tasks pulls no wan_va.
"""
from __future__ import annotations

from ptqeval.wam.lingbot_va.tasks import (  # noqa: F401
    ALL_TASKS,
    EVAL_STEP_LIMIT,
    SELECTED_15_TASKS,
    SMOKE_5_TASKS,
)

__all__ = ["ALL_TASKS", "SELECTED_15_TASKS", "SMOKE_5_TASKS", "EVAL_STEP_LIMIT"]
