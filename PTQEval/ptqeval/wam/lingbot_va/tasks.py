# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Selected RoboTwin task list and step limits for the lingbot-va WAM.

15-task subset: 1 long + 4 medium-long + 10 short. Used by
ptqeval/eval/run_eval.sh via:
    python -c "from ptqeval.wam.lingbot_va.tasks import SELECTED_15_TASKS \\
               as t; print(' '.join(t))"

EVAL_STEP_LIMIT mirrors RoboTwin/task_config/_eval_step_limit.yml for the
subset; kept here so cross-WAM aggregators can reason about wall-clock
without parsing the RoboTwin yaml.
"""
from __future__ import annotations

SELECTED_15_TASKS: list[str] = [
    "put_bottles_dustbin",                              # long (1700)
    "hanging_mug", "stack_bowls_two",                   # medium-long (900)
    "handover_block",                                   # medium-long (800)
    "place_can_basket",                                 # medium-long (700)
    "place_empty_cup", "place_shoe", "place_bread_skillet",  # short (500)
    "adjust_bottle", "beat_block_hammer", "click_bell",
    "lift_pot", "place_a2b_left", "pick_dual_bottles",
    "turn_switch",                                      # short (400 x7)
]

EVAL_STEP_LIMIT: dict[str, int] = {
    "put_bottles_dustbin": 1700,
    "hanging_mug": 900,
    "stack_bowls_two": 900,
    "handover_block": 800,
    "place_can_basket": 700,
    "place_empty_cup": 500,
    "place_shoe": 500,
    "place_bread_skillet": 500,
    "adjust_bottle": 400,
    "beat_block_hammer": 400,
    "click_bell": 400,
    "lift_pot": 400,
    "place_a2b_left": 400,
    "pick_dual_bottles": 400,
    "turn_switch": 400,
}
