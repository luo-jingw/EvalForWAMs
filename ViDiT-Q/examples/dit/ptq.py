"""
conduct model PTQ process
take in the orginal model and the calib data (optional)
save the quantized model checkpoint
"""
import torch
import sys
import os
import logging
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
import numpy as np
from omegaconf import OmegaConf
from models.models import DiT,DiT_models
from models.download import find_model
import torch.nn as nn
import torch.nn.functional as F

from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from models.quant_dit import QuantDiT


def main(args):

    # PTQ main function:

    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    latent_size = args.image_size // 8
    ptq_config_file = args.ptq_config

    latent_size = args.image_size // 8
    ptq_config_file = args.ptq_config
    quant_config = OmegaConf.load(ptq_config_file)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model=QuantDiT(quant_config,
     ckpt_path,
     depth=28,
     hidden_size=1152, 
     patch_size=2, 
     num_heads=16, 
     input_size=latent_size,
     num_classes=args.num_classes,
     ).to(device)
    
    model.half()
    model.eval()  # INFO: make sure to set the model into eval mode
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [217, 363, 347, 574, 188, 99, 47, 379]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    '''
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''

    '''
    INFO: the smooth_quant quantization.
    load act channel mask from the calib data
    '''
    if quant_config.get("smooth_quant",None) is not None:
        # INFO: the SQQuantizedLayer are initialized with the quant_layer_refactor_ in quant_dit.py

        def init_sq_channel_mask_(module, full_name, calib_data):
            assert isinstance(module, SQQuantizedLinear)
            act_mask = calib_data[full_name].mean(dim=0)  # [T, C], averaged over all timesteps
            module.get_channel_mask(act_mask)  # set self.channel_mask
            module.update_quantized_weight_scaled()

        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(os.path.join(args.log, quant_config.calib_data.save_path), weights_only=True)  # default wtih weights_only=True, will cause warning

        # get the channel mask, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=SQQuantizedLinear,  # add hook to all objects of this cls
                            function=init_sq_channel_mask_,
                            calib_data = calib_data,
                            full_name='',
                            **kwargs
                            )

    '''
    INFO: the quarot quantization.
    init and apply the rotation matrix
    '''
    if quant_config.get("quarot",None) is not None:

        def init_rotation_matrix_(module, full_name):
            from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
            module.get_rotation_matrix()
            module.update_quantized_weight_rotated()

        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        # get the rotation matrix, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=QuarotQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_matrix_,
                            full_name='',
                            **kwargs
                            )

    model.set_init_done()
    model.save_quant_param_dict()
    torch.save(model.quant_param_dict, os.path.join(args.log, 'quant_params.pth'))

    # Test with model inference
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    t = torch.tensor([1] * z.shape[0], device=device)
    _ = model(z,t,y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument('--ptq-config', default='./configs/w8a8.yaml', type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
