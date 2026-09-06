#!/usr/bin/env python3
"""Gradio control panel for the G1 ImageWAM policy server.

One process, one resident model. A WebSocket policy server (for the robot client)
runs in a daemon thread; this Gradio app runs on the main thread and drives the same
`PolicyEngine` live -- so you can switch checkpoint, inference method (baseline /
ProbeFlow / DASH) and per-method config from a browser without restarting anything or
disturbing the connected robot beyond the brief stall of a checkpoint reload.

    ssh nnmc75
    cd /home/user1/workspace/jingwu/ImagewamFT/ImageWAM
    set -a && source .env.local && set +a
    CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/g1/gradio_control_panel.py \
        --g1-client-path /home/user1/workspace/jingwu/ImagewamFT/g1-client \
        --runs-root runs \
        --prompt-embeds-dir ../checkpoints/imagewam_g1 \
        --initial-profile g1_stack_cubes_flux2_klein_4b \
        --ws-port 8000 --gradio-port 7860 \
        --auth-user g1 --auth-pass <choose-a-password>

The WebSocket robot server listens on --ws-port (unchanged wire protocol). The Gradio
UI is exposed publicly via a share link (gradio.live) protected by basic auth.

torch.compile is never enabled on the inference path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gradio as gr

from inference_methods import METHODS, BaselineParams, DashParams, ProbeFlowParams
from serve_imagewam_g1 import (
    ActiveProfile,
    RuntimeConfig,
    build_engine,
    load_wire_codec,
    serve,
)

log = logging.getLogger("g1_control_panel")

# Known run-dir -> prompt-embeds filename. Anything not listed falls back to a guess
# and, failing that, the user pastes a path in the UI.
PROMPT_EMBEDS_BY_TASK = {
    "g1_flux2_klein_4b_base_imagewam": "g1_prompt_embeds.npz",
    "g1_stack_cubes_flux2_klein_4b": "g1_stack_cubes_prompt_embeds.npz",
    "g1_pick_red_bottle_flux2_klein_4b": "g1_pick_red_bottle_prompt_embeds.npz",
    "g1_sort_tool_flux2_klein_4b": "g1_sort_tool_prompt_embeds.npz",
}


@dataclass
class RunProfile:
    """A discovered training run: fixes task/stats/prompt-embeds; holds its step list.

    `prompt_embeds` is the *intended* npz path (used for both loading and persisting);
    `embeds_exists` says whether it is on disk yet. A missing npz can still be loaded
    (empty bank) and bootstrapped from the panel's Prompt precompute section.
    """

    task: str
    timestamp: str
    run_dir: Path
    dataset_stats: Path
    prompt_embeds: Path
    embeds_exists: bool
    steps: list[tuple[str, Path]]  # (label, weights path), newest last

    @property
    def name(self) -> str:
        return f"{self.task}/{self.timestamp}"


def discover_profiles(runs_root: Path, prompt_embeds_dir: Path) -> list[RunProfile]:
    """Scan runs/<task>/<timestamp>/ for runs that have both weights and stats."""
    profiles: list[RunProfile] = []
    if not runs_root.exists():
        log.warning("runs root %s does not exist", runs_root)
        return profiles
    for task_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for ts_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            stats = ts_dir / "dataset_stats.json"
            weights_dir = ts_dir / "checkpoints" / "weights"
            if not stats.exists() or not weights_dir.exists():
                continue
            steps = sorted(weights_dir.glob("step_*.pt"), key=lambda p: _step_num(p))
            if not steps:
                continue
            embeds_name = PROMPT_EMBEDS_BY_TASK.get(task_dir.name, f"{task_dir.name}_prompt_embeds.npz")
            embeds = prompt_embeds_dir / embeds_name
            profiles.append(
                RunProfile(
                    task=task_dir.name,
                    timestamp=ts_dir.name,
                    run_dir=ts_dir,
                    dataset_stats=stats,
                    prompt_embeds=embeds,
                    embeds_exists=embeds.exists(),
                    steps=[(p.stem, p) for p in steps],
                )
            )
    return profiles


def _step_num(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


class ControlPanel:
    """Binds the Gradio widgets to a running PolicyEngine + profile registry."""

    def __init__(self, engine, profiles: list[RunProfile]) -> None:
        self._engine = engine
        self._profiles = {p.name: p for p in profiles}

    # -- helpers ----------------------------------------------------------- #
    def _profile_names(self) -> list[str]:
        return list(self._profiles)

    def _status_text(self) -> str:
        snap = self._engine.snapshot()
        timing = self._engine.last_timing
        if timing is not None:
            snap["last_timing_ms"] = {
                "preprocess": round(timing.preprocess_s * 1e3, 1),
                "infer": round(timing.infer_s * 1e3, 1),
                "postprocess": round(timing.postprocess_s * 1e3, 1),
                "effective_steps": timing.effective_steps,
                "dash": timing.dash,
            }
        return json.dumps(snap, indent=2, default=str)

    def _fm_stats_line(self) -> str:
        stats = self._engine.snapshot().get("fm_step_stats", {}) or {}
        if not stats.get("n"):
            return ""
        return (
            f"\n\n**FM step 统计（本配置累计 n={stats['n']}）**："
            f"均值 **{stats['mean']}** ｜ 方差 **{stats['var']}** ｜ 标准差 {stats['std']} "
            f"｜ min {stats['min']} / max {stats['max']}"
            f"　_（Apply 配置 / 换 checkpoint / 点下方 Reset 会清零）_"
        )

    def _dash_readout(self) -> str:
        timing = self._engine.last_timing
        stats_line = self._fm_stats_line()
        if timing is None or timing.dash is None:
            return "**DASH FM step**：切到 DASH 方法、机器人跑一帧后点 Refresh 查看。" + stats_line
        d = timing.dash
        head = (
            f"**DASH 上次 replan** ｜ 有效 **FM steps = {d.get('parallel_calls', '?')}** "
            f"(标称 num_inference_steps=10) ｜ 跳步 k: {d.get('base_step', '?')} → 接受到 "
            f"{d.get('chosen_step', '?')} ｜ accepted_links={d.get('accepted_links', '?')} "
            f"｜ drift={d.get('drift', '?')} ｜ planned_k={d.get('planned_step_k', '?')}"
        )
        return head + stats_line

    def _on_refresh(self):
        return self._status_text(), self._dash_readout()

    def on_reset_fm_stats(self):
        self._engine.reset_fm_stats()
        return self._status_text(), self._dash_readout()

    # -- callbacks --------------------------------------------------------- #
    def on_profile_change(self, profile_name: str):
        profile = self._profiles.get(profile_name)
        if profile is None:
            return gr.update(), "", "unknown profile"
        step_labels = [label for label, _ in profile.steps]
        latest = step_labels[-1] if step_labels else None
        embeds = str(profile.prompt_embeds)
        warn = "" if profile.embeds_exists else (
            f"⚠ 该任务 '{profile.task}' 还没有 prompt-embeds npz（{profile.prompt_embeds.name}）。"
            "可以直接 Load（空库），再到下方 Prompt 预计算加该任务指令，会自动创建这个 npz。"
        )
        return (
            gr.update(choices=step_labels, value=latest),
            embeds,
            warn,
        )

    def on_load_checkpoint(self, profile_name: str, step_label: str, embeds_path: str):
        profile = self._profiles.get(profile_name)
        if profile is None:
            return "error: unknown profile", self._status_text(), gr.update()
        step_map = dict(profile.steps)
        if step_label not in step_map:
            return f"error: step {step_label!r} not in {profile.name}", self._status_text(), gr.update()
        # A missing npz is allowed: the engine starts an empty bank; bootstrap it via
        # the Prompt precompute section (add prompts -> writes this npz).
        embeds = Path(embeds_path).expanduser() if embeds_path.strip() else profile.prompt_embeds
        active = ActiveProfile(
            name=f"{profile.name}:{step_label}",
            ckpt_path=step_map[step_label],
            dataset_stats_path=profile.dataset_stats,
            prompt_embeds_path=Path(embeds),
            task_config=profile.task,
        )
        try:
            self._engine.swap_checkpoint(active)
        except Exception as error:  # noqa: BLE001 - surface to the UI
            return f"load failed: {error}", self._status_text(), gr.update()
        note = "" if Path(embeds).exists() else "（空 prompt 库：先到下方 Prompt 预计算加指令）"
        # new bank -> refresh override choices, clear any previous override
        return (f"✓ loaded {active.name} {note}", self._status_text(),
                gr.update(choices=self._prompt_choices(), value=""))

    def on_apply_config(
        self,
        method: str,
        action_horizon: int,
        exec_horizon: int,
        base_steps: int,
        base_seed: str,
        pf_steps: int,
        pf_dt_probe: float,
        pf_epsilon: float,
        pf_n_min: int,
        pf_n_max: int,
        pf_delta_n: int,
        dash_steps: int,
        dash_k_near: int,
        dash_k_far: int,
        dash_drift_low: float,
        dash_drift_high: float,
        dash_verify_rel_l2_max: float,
        dash_speculative: bool,
    ):
        seed = int(base_seed) if str(base_seed).strip() not in ("", "None") else None
        try:
            self._engine.update_runtime(
                method=method,
                action_horizon=int(action_horizon),
                exec_horizon=int(exec_horizon),
                baseline={"num_inference_steps": int(base_steps), "seed": seed},
                probeflow={
                    "num_inference_steps": int(pf_steps),
                    "probeflow_dt_probe": float(pf_dt_probe),
                    "probeflow_epsilon": float(pf_epsilon),
                    "probeflow_n_min": int(pf_n_min),
                    "probeflow_n_max": int(pf_n_max),
                    "probeflow_delta_n": int(pf_delta_n),
                },
                dash={
                    "num_inference_steps": int(dash_steps),
                    "spec_ratio_jump_k_near": int(dash_k_near),
                    "spec_ratio_jump_k_far": int(dash_k_far),
                    "spec_ratio_jump_drift_low": float(dash_drift_low),
                    "spec_ratio_jump_drift_high": float(dash_drift_high),
                    "spec_ratio_jump_verify_rel_l2_max": float(dash_verify_rel_l2_max),
                    "spec_ratio_jump_speculative": bool(dash_speculative),
                },
            )
        except Exception as error:  # noqa: BLE001
            return f"apply failed: {error}", self._status_text()
        return f"✓ method={method}", self._status_text()

    def _prompt_choices(self) -> list[str]:
        # "" = don't override (use whatever prompt the robot client sends).
        return [""] + list(self._engine.prompts)

    def on_set_prompt(self, prompt: str):
        try:
            self._engine.set_prompt_override(prompt or None)
        except Exception as error:  # noqa: BLE001
            return f"设置失败: {error}", self._status_text()
        active = prompt if prompt else "（用客户端发来的 prompt）"
        return f"✓ 当前推理用 prompt: {active}", self._status_text()

    def on_add_prompt(self, text: str, persist: bool):
        try:
            r = self._engine.add_prompt(text, persist=bool(persist))
        except Exception as error:  # noqa: BLE001
            return f"add failed: {error}", self._status_text(), gr.update()
        label = {
            "added": ("✓ 已加入并保存 npz" if persist else "✓ 已加入（仅内存，未存盘）"),
            "exists": "已在库里，无需重复计算",
            "empty": "空 prompt",
        }.get(r.get("status", ""), r.get("status", ""))
        # refresh the override dropdown so the new prompt is selectable
        return f"{label}: {r.get('prompt', '')}", self._status_text(), gr.update(choices=self._prompt_choices())

    def on_toggle_auto(self, enabled: bool):
        self._engine.set_auto_precompute(bool(enabled))
        return f"机器人发来未知 prompt 时自动预计算: {'ON' if enabled else 'OFF'}"

    def build(self) -> gr.Blocks:
        base = BaselineParams()
        pf = ProbeFlowParams()
        dash = DashParams()
        names = self._profile_names()
        init = names[0] if names else None
        init_steps = [l for l, _ in self._profiles[init].steps] if init else []

        with gr.Blocks(title="G1 ImageWAM control panel") as demo:
            gr.Markdown("# G1 ImageWAM — inference control panel")
            gr.Markdown(
                "Switch checkpoint and inference method (baseline / ProbeFlow / DASH) live. "
                "The robot client stays connected to the WebSocket server; changes take effect "
                "on its next chunk. A checkpoint reload briefly stalls inference."
            )

            with gr.Row():
                profile_dd = gr.Dropdown(names, value=init, label="Run (task / timestamp)")
                step_dd = gr.Dropdown(init_steps, value=(init_steps[-1] if init_steps else None), label="Checkpoint step")
                embeds_tb = gr.Textbox(label="Prompt-embeds npz", scale=2)
            with gr.Row():
                load_btn = gr.Button("Load checkpoint", variant="primary")
                load_status = gr.Markdown()

            gr.Markdown("### Inference method & config")
            method_radio = gr.Radio(list(METHODS), value="baseline", label="Method")
            with gr.Row():
                action_horizon = gr.Number(value=16, label="action_horizon (chunk len)", precision=0)
                exec_horizon = gr.Number(value=16, label="exec_horizon (advisory, client)", precision=0)

            with gr.Accordion("baseline params", open=True):
                with gr.Row():
                    base_steps = gr.Number(value=base.num_inference_steps, label="num_inference_steps", precision=0)
                    base_seed = gr.Textbox(value="None", label="seed (int or None)")
            with gr.Accordion("ProbeFlow params", open=False):
                with gr.Row():
                    pf_steps = gr.Number(value=pf.num_inference_steps, label="num_inference_steps", precision=0)
                    pf_dt_probe = gr.Number(value=pf.probeflow_dt_probe, label="dt_probe")
                    pf_epsilon = gr.Number(value=pf.probeflow_epsilon, label="epsilon")
                with gr.Row():
                    pf_n_min = gr.Number(value=pf.probeflow_n_min, label="n_min", precision=0)
                    pf_n_max = gr.Number(value=pf.probeflow_n_max, label="n_max", precision=0)
                    pf_delta_n = gr.Number(value=pf.probeflow_delta_n, label="delta_n", precision=0)
            with gr.Accordion("DASH params", open=False):
                with gr.Row():
                    dash_steps = gr.Number(value=dash.num_inference_steps, label="num_inference_steps", precision=0)
                    dash_k_near = gr.Number(value=dash.spec_ratio_jump_k_near, label="k_near", precision=0)
                    dash_k_far = gr.Number(value=dash.spec_ratio_jump_k_far, label="k_far (= N-1)", precision=0)
                with gr.Row():
                    dash_drift_low = gr.Number(value=dash.spec_ratio_jump_drift_low, label="drift_low")
                    dash_drift_high = gr.Number(value=dash.spec_ratio_jump_drift_high, label="drift_high")
                    dash_verify = gr.Number(value=dash.spec_ratio_jump_verify_rel_l2_max, label="verify_rel_l2_max")
                    dash_spec = gr.Checkbox(value=dash.spec_ratio_jump_speculative, label="speculative")

            with gr.Row():
                apply_btn = gr.Button("Apply config", variant="primary")
                apply_status = gr.Markdown()

            gr.Markdown("### Prompt（切换推理用的指令）")
            gr.Markdown(
                "选一条库里的 prompt，服务端就用它推理、忽略机器人客户端发来的那条；"
                "留空 = 用客户端发来的 prompt。切换立即生效，不用重启客户端。"
            )
            with gr.Row():
                prompt_dd = gr.Dropdown(self._prompt_choices(), value="",
                                        label="Prompt override（空=用客户端的）", scale=3)
                prompt_status = gr.Markdown()

            gr.Markdown("### Prompt 预计算")
            gr.Markdown(
                "服务端不常驻文本编码器。这里现算一个新 prompt（首次会加载 Qwen3 ~8GB、卡一下），"
                "用模型自带的文本编码路径，和训练/离线 npz 完全一致，可选存回该 run 的 npz。"
            )
            with gr.Row():
                new_prompt = gr.Textbox(label="新 prompt", scale=3)
                persist_ck = gr.Checkbox(value=True, label="保存到 npz")
                add_prompt_btn = gr.Button("Precompute & add", variant="primary")
            add_prompt_status = gr.Markdown()
            auto_ck = gr.Checkbox(
                value=False,
                label="机器人发来未知 prompt 时自动预计算（首次会卡一下加载 Qwen3；默认关）",
            )

            gr.Markdown("### Status")
            with gr.Row():
                refresh_btn = gr.Button("Refresh status / latency / DASH FM step")
                reset_fm_btn = gr.Button("Reset FM stats")
            dash_readout = gr.Markdown(self._dash_readout())
            status_box = gr.Code(value=self._status_text(), language="json", label="engine state")

            # wiring
            profile_dd.change(self.on_profile_change, [profile_dd], [step_dd, embeds_tb, load_status])
            load_btn.click(self.on_load_checkpoint, [profile_dd, step_dd, embeds_tb],
                           [load_status, status_box, prompt_dd])
            prompt_dd.change(self.on_set_prompt, [prompt_dd], [prompt_status, status_box])
            apply_btn.click(
                self.on_apply_config,
                [method_radio, action_horizon, exec_horizon, base_steps, base_seed,
                 pf_steps, pf_dt_probe, pf_epsilon, pf_n_min, pf_n_max, pf_delta_n,
                 dash_steps, dash_k_near, dash_k_far, dash_drift_low, dash_drift_high,
                 dash_verify, dash_spec],
                [apply_status, status_box],
            )
            refresh_btn.click(self._on_refresh, None, [status_box, dash_readout])
            reset_fm_btn.click(self.on_reset_fm_stats, None, [status_box, dash_readout])
            add_prompt_btn.click(self.on_add_prompt, [new_prompt, persist_ck],
                                 [add_prompt_status, status_box, prompt_dd])
            auto_ck.change(self.on_toggle_auto, [auto_ck], [add_prompt_status])
        return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-client-path", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--prompt-embeds-dir", type=Path, default=Path("../checkpoints/imagewam_g1"))
    parser.add_argument("--initial-profile", default=None,
                        help="task name (or task/timestamp) to load at startup; default = newest discovered")
    parser.add_argument("--flux2-model-path", type=Path, default=os.environ.get("FLUX2_MODEL_PATH"))
    parser.add_argument("--ae-model-path", type=Path, default=os.environ.get("FLUX2_AE_MODEL_PATH"))
    parser.add_argument("--flux2-src-path", type=Path, default=os.environ.get("FLUX2_SRC"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robotwin-camera-layout", default="compact_288x256")
    parser.add_argument("--ws-host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=8000)
    parser.add_argument("--gradio-port", type=int, default=7860)
    parser.add_argument("--auth-user", default=os.environ.get("GRADIO_USER"))
    parser.add_argument("--auth-pass", default=os.environ.get("GRADIO_PASS"))
    parser.add_argument("--no-share", action="store_true", help="bind locally instead of a public share link")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for name, value in (
        ("--flux2-model-path/FLUX2_MODEL_PATH", args.flux2_model_path),
        ("--ae-model-path/FLUX2_AE_MODEL_PATH", args.ae_model_path),
        ("--flux2-src-path/FLUX2_SRC", args.flux2_src_path),
    ):
        if value is None:
            raise ValueError(f"{name} is required (set it or source .env.local)")

    profiles = discover_profiles(args.runs_root, args.prompt_embeds_dir)
    if not profiles:
        raise SystemExit(f"no runs discovered under {args.runs_root}")
    log.info("discovered %d profiles: %s", len(profiles), [p.name for p in profiles])

    # Pick the initial profile: match by task name (newest timestamp) or task/timestamp;
    # default prefers a profile that already has its prompt-embeds npz.
    if args.initial_profile:
        matches = [p for p in profiles if p.name == args.initial_profile or p.task == args.initial_profile]
        if not matches:
            raise SystemExit(f"--initial-profile {args.initial_profile!r} matched none of {[p.name for p in profiles]}")
        initial = sorted(matches, key=lambda p: p.timestamp)[-1]
    else:
        pool = [p for p in profiles if p.embeds_exists] or profiles
        initial = sorted(pool, key=lambda p: (p.task, p.timestamp))[-1]
    if not initial.embeds_exists:
        log.warning("initial profile %s has no prompt-embeds npz -> empty bank; add prompts via the panel",
                    initial.name)
    latest_step_label, latest_step_path = initial.steps[-1]
    log.info("initial profile: %s @ %s", initial.name, latest_step_label)

    codec = load_wire_codec(args.g1_client_path)
    engine = build_engine(
        ckpt=latest_step_path,
        dataset_stats=initial.dataset_stats,
        prompt_embeds=initial.prompt_embeds,
        task_config=initial.task,
        flux2_model_path=args.flux2_model_path,
        ae_model_path=args.ae_model_path,
        flux2_src_path=args.flux2_src_path,
        device=args.device,
        camera_layout=args.robotwin_camera_layout,
        host=args.ws_host,
        port=args.ws_port,
        runtime=RuntimeConfig(),
        profile_name=f"{initial.name}:{latest_step_label}",
    )
    log.info("engine ready: %s", engine.metadata())

    # WebSocket robot server in a daemon thread; Gradio owns the main thread.
    ready = threading.Event()
    ws_thread = threading.Thread(
        target=serve, args=(engine, args.ws_host, args.ws_port, codec, ready), daemon=True, name="ws-serve"
    )
    ws_thread.start()
    ready.wait(timeout=30)
    log.info("WebSocket policy server up on ws://%s:%d", args.ws_host, args.ws_port)

    panel = ControlPanel(engine, profiles)
    demo = panel.build()

    share = not args.no_share
    # Auth is optional: pass --auth-user/--auth-pass to gate the public link, or omit
    # for a no-login share link.
    auth = (args.auth_user, args.auth_pass) if (args.auth_user and args.auth_pass) else None
    if share and auth is None:
        log.warning("public share link has NO login -- anyone with the URL can load models / use the GPU")
    demo.launch(
        server_name="0.0.0.0" if not share else "127.0.0.1",
        server_port=args.gradio_port,
        share=share,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
