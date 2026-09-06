# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Real-robot deployment config for Unitree G1 (websocket server mode).
from easydict import EasyDict
from .va_g1_train_cfg import va_g1_train_cfg

va_g1_server_cfg = EasyDict(__name__='Config: VA g1 server')
va_g1_server_cfg.update(va_g1_train_cfg)

# point to the assembled inference model dir (trained transformer + base vae/tokenizer/text_encoder)
va_g1_server_cfg.wan22_pretrained_model_name_or_path = "/shared/user64/workspace/yuhao/lingbot-va/checkpoints/red-1000"

va_g1_server_cfg.infer_mode = 'server'     # websocket server; obs come from the robot client
va_g1_server_cfg.enable_offload = True     # offload vae & text_encoder to CPU to save VRAM
va_g1_server_cfg.host = '0.0.0.0'
va_g1_server_cfg.port = 29536

# sampling params
va_g1_server_cfg.guidance_scale = 5
va_g1_server_cfg.action_guidance_scale = 1
va_g1_server_cfg.num_inference_steps = 20
va_g1_server_cfg.video_exec_step = -1
va_g1_server_cfg.action_num_inference_steps = 50
