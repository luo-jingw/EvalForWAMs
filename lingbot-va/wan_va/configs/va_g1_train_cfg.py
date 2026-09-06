# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Post-training config for Unitree G1 (16-dim joint action, no EEF).
import os
from easydict import EasyDict
from .shared_config import va_shared_cfg

va_g1_train_cfg = EasyDict(__name__='Config: VA g1 train')
va_g1_train_cfg.update(va_shared_cfg)

# --- paths (edit these) ---
va_g1_train_cfg.wan22_pretrained_model_name_or_path = "/home/bioprocessing-lab/yuhao/lingbot-va/model/base"
va_g1_train_cfg.dataset_path = "/home/bioprocessing-lab/yuhao/lingbot-va/data/red-bottle"  # dir holding meta/ data/ videos/ latents/
va_g1_train_cfg.empty_emb_path = os.path.join(va_g1_train_cfg.dataset_path, 'empty_emb.pt')

# --- video / latent layout ---
va_g1_train_cfg.env_type = 'none'        # joints -> skip robotwin relative-pose; cams concat on width
va_g1_train_cfg.attn_window = 30
va_g1_train_cfg.frame_chunk_size = 4
va_g1_train_cfg.height = 256
va_g1_train_cfg.width = 256
va_g1_train_cfg.obs_cam_keys = [
    'observation.images.cam_left_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist',
]

# --- action space ---
# raw 16 dims = [L joints 0-6, R joints 7-13, L grip 14, R grip 15]
# mapped into the standard 30-dim slots: L-joints 14-20, R-joints 21-27, L-grip 28, R-grip 29
va_g1_train_cfg.action_dim = 30
va_g1_train_cfg.action_per_frame = 8     # = 4 (VAE temporal factor) * frame stride (2 @ fps15 on 30fps source)
va_g1_train_cfg.used_action_channel_ids = list(range(14, 28)) + [28, 29]
inverse = [len(va_g1_train_cfg.used_action_channel_ids)] * va_g1_train_cfg.action_dim
for i, j in enumerate(va_g1_train_cfg.used_action_channel_ids):
    inverse[j] = i
va_g1_train_cfg.inverse_used_action_channel_ids = inverse

va_g1_train_cfg.snr_shift = 5.0
va_g1_train_cfg.action_snr_shift = 0.05
va_g1_train_cfg.action_norm_method = 'quantiles'
# 30-dim, standard order. Fill slots 14-29 from compute_g1_action_stats.py; EEF slots 0-13 stay 0.
va_g1_train_cfg.norm_stat = {
    "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9318013787269592, -0.12740090489387512, -0.573829174041748, -0.8592005372047424, -0.981776237487793, -0.5886626839637756, -1.3971589803695679, -1.0969691276550293, -0.8614471554756165, -0.19088001549243927, -0.8153378367424011, -0.15571367740631104, -0.5743945240974426, -0.2264183610677719, 1.649195671081543, 1.6187511682510376],
    "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6146520376205444, 0.7885221838951111, 0.12712091207504272, 0.8583471775054932, 0.24922075867652893, 1.6143178939819336, 0.5484979152679443, 0.6596460938453674, 0.09389922022819519, 0.6977345943450928, 1.0046262741088867, 0.7062154412269592, 1.614305019378662, 1.4045097827911377, 5.400000095367432, 5.400000095367432],
}

# --- training ---
va_g1_train_cfg.enable_wandb = False
va_g1_train_cfg.load_worker = 16
va_g1_train_cfg.save_interval = 500
va_g1_train_cfg.gc_interval = 50
va_g1_train_cfg.cfg_prob = 0.1
va_g1_train_cfg.learning_rate = 1e-5
va_g1_train_cfg.beta1 = 0.9
va_g1_train_cfg.beta2 = 0.95
va_g1_train_cfg.weight_decay = 1e-1
va_g1_train_cfg.warmup_steps = 10
va_g1_train_cfg.batch_size = 1
va_g1_train_cfg.gradient_accumulation_steps = 10
va_g1_train_cfg.num_steps = 3000
