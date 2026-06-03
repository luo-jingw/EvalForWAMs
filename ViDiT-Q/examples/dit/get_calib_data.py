"""
adapted from the orginal fp_inference script
add the hook and the save of the activation data
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

from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

class SaveActivationHook:

    def __init__(self):
        self.hook_handle = None
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        '''
        the input shape could be [BS, C] or [BS, N_token, C]
        only keep the channel_dim for reduced saved act size
        '''
        C = module_in[0].shape[-1]
        data = module_in[0].reshape([-1,C]).abs().max(dim=0)[0]  # [C]

        self.outputs.append(data)

    def clear(self):
        self.outputs = []

def add_hook_to_module_(module, hook_cls):
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook

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
    quant_config = OmegaConf.load(ptq_config_file)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model=DiT(
     input_size=latent_size,
     patch_size=2, 
     in_channels=4,
     hidden_size=1152, 
     depth=28,
     num_heads=16, 
     num_classes=args.num_classes,
     ).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    '''
    INFO: add the hook for hooking the activations
    '''
    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(model,
                            class_type=nn.Linear,  # add hook to all objects of this cls
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [217, 363, 347, 574, 188, 99, 47, 379]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y)
    t = torch.tensor([1] * z.shape[0], device=device)
    _=model(z,y,t)
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, "fp_sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    save_d = {}
    for k,v in hook_d.items():
        save_d[k] = torch.stack(v.outputs, dim=0)  # [N_timestep, C]
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()

    torch.save(save_d, os.path.join(args.log, quant_config.calib_data.save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument('--ptq-config', default='./configs/config.yaml', type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant_param_ckpt", type=str, default="./quant_params.pth")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
