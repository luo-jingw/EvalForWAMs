# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Image-to-Video-Action inference config for Unitree G1.
from easydict import EasyDict
from .va_g1_train_cfg import va_g1_train_cfg

va_g1_i2va_cfg = EasyDict(__name__='Config: VA g1 i2va')
va_g1_i2va_cfg.update(va_g1_train_cfg)

# point to the assembled inference model dir (trained transformer + base vae/tokenizer/text_encoder)
va_g1_i2va_cfg.wan22_pretrained_model_name_or_path = "/home/bioprocessing-lab/yuhao/lingbot-va/model/g1_infer"

va_g1_i2va_cfg.infer_mode = 'i2va'
va_g1_i2va_cfg.enable_offload = True       # offload vae & text_encoder to CPU to save VRAM

# inference inputs: one <cam_key>.png per obs_cam_keys under this dir
va_g1_i2va_cfg.input_img_path = 'example/g1'
va_g1_i2va_cfg.prompt = 'pick red bottle'
va_g1_i2va_cfg.num_chunks_to_infer = 10

# sampling params
va_g1_i2va_cfg.guidance_scale = 5
va_g1_i2va_cfg.action_guidance_scale = 1
va_g1_i2va_cfg.num_inference_steps = 20
va_g1_i2va_cfg.video_exec_step = -1
va_g1_i2va_cfg.action_num_inference_steps = 50
