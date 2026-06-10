# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""RoboTwin task lists and step limits for the lingbot-va WAM.

Two task lists:
  SELECTED_15_TASKS    -- production eval subset (1 long + 4 medium-long
                          + 10 short). Used by run_eval.sh's pool/single
                          modes by default.
  CALIB_TASKS_ALL      -- all 50 RoboTwin tasks. Used by Phase 31
                          calibration data collection (5 ep / task) so
                          per-channel activation absmax covers the full
                          task-space diversity for SmoothQuant
                          channel_mask and static activation scales.

EVAL_STEP_LIMIT mirrors RoboTwin/task_config/_eval_step_limit.yml for
all 50 tasks; cross-WAM aggregators can reason about wall-clock without
parsing the RoboTwin yaml.

CLI access:
    python -c "from ptqeval.wam.lingbot_va.tasks import SELECTED_15_TASKS \\
               as t; print(' '.join(t))"
    python -c "from ptqeval.wam.lingbot_va.tasks import CALIB_TASKS_ALL \\
               as t; print(' '.join(t))"
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

# Phase 31 (v2): all 50 RoboTwin tasks for broader activation statistics.
# Sourced from RoboTwin/task_config/_eval_step_limit.yml. Calibration runs
# this list with 5 ep/task -> 250 trajectories worth of per-channel
# absmax aggregation (running-max merged via _CalibState; sequential
# single-mode invocation in the calib runner avoids cross-process race
# on calib_data.pth).
CALIB_TASKS_ALL: list[str] = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
    "blocks_ranking_size", "click_alarmclock", "click_bell",
    "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic",
    "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
    "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch",
]
assert len(CALIB_TASKS_ALL) == 50, f"expected 50, got {len(CALIB_TASKS_ALL)}"
assert set(SELECTED_15_TASKS).issubset(CALIB_TASKS_ALL), (
    "SELECTED_15_TASKS must be a subset of CALIB_TASKS_ALL"
)

EVAL_STEP_LIMIT: dict[str, int] = {
    "adjust_bottle": 400, "beat_block_hammer": 400,
    "blocks_ranking_rgb": 1200, "blocks_ranking_size": 1200,
    "click_alarmclock": 400, "click_bell": 400, "dump_bin_bigbin": 600,
    "grab_roller": 400, "handover_block": 800, "handover_mic": 600,
    "hanging_mug": 900, "lift_pot": 400, "move_can_pot": 400,
    "move_pillbottle_pad": 400, "move_playingcard_away": 400,
    "move_stapler_pad": 400, "open_laptop": 700, "open_microwave": 1500,
    "pick_diverse_bottles": 400, "pick_dual_bottles": 400,
    "place_a2b_left": 400, "place_a2b_right": 400,
    "place_bread_basket": 700, "place_bread_skillet": 500,
    "place_burger_fries": 500, "place_can_basket": 700,
    "place_cans_plasticbox": 800, "place_container_plate": 400,
    "place_dual_shoes": 600, "place_empty_cup": 500, "place_fan": 400,
    "place_mouse_pad": 400, "place_object_basket": 700,
    "place_object_scale": 400, "place_object_stand": 400,
    "place_phone_stand": 400, "place_shoe": 500, "press_stapler": 400,
    "put_bottles_dustbin": 1700, "put_object_cabinet": 700,
    "rotate_qrcode": 400, "scan_object": 500, "shake_bottle": 700,
    "shake_bottle_horizontally": 700, "stack_blocks_three": 1200,
    "stack_blocks_two": 800, "stack_bowls_three": 1200,
    "stack_bowls_two": 900, "stamp_seal": 400, "turn_switch": 400,
}
assert set(EVAL_STEP_LIMIT.keys()) == set(CALIB_TASKS_ALL), (
    "EVAL_STEP_LIMIT must cover exactly the 50 calib tasks"
)
