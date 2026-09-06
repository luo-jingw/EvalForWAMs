"""RoboTwin task lists for the imagewam WAM.

The RoboTwin task set is benchmark data shared across the WAMs. Re-exported from
the lingbot_va list to keep a single source of truth (importing lingbot_va.tasks
pulls no wan_va). NOTE: verify ImageWAM's vendored RoboTwin task set matches when
the env is up; if it diverges, read ImageWAM/third_party/RoboTwin/task_config/
_eval_step_limit.yml instead.
"""
from __future__ import annotations

from ptqeval.wam.lingbot_va.tasks import (  # noqa: F401
    ALL_TASKS,
    EVAL_STEP_LIMIT,
    SELECTED_15_TASKS,
    SMOKE_5_TASKS,
)

__all__ = ["ALL_TASKS", "SELECTED_15_TASKS", "SMOKE_5_TASKS", "EVAL_STEP_LIMIT"]
