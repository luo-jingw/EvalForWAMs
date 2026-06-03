"""
follow the orginial model inference script, 
and add the following parts. 
run the model with algorithm-level quantization simulation
"""

"""
conduct model PTQ process
take in the orginal model and the calib data
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
from omegaconf import ListConfig
from models.models import DiT,DiT_models
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from models.quant_dit import QuantDiT
from models.download import find_model

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
    model=QuantDiT(quant_config,
     ckpt_path,
     depth=28,
     hidden_size=1152,
     patch_size=2,
     num_heads=16,
     input_size=latent_size,
     num_classes=args.num_classes,
     ).to(device)

    model.half()   # use FP16
    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()

    if args.hardware:  # use the cuda kernel
        assert not if_mixed_precision, ("mixed precision is currently not supported in CUDA kernels")
        if args.quant_weight_ckpt is None:
            save_path = os.path.join(args.log, 'int_weight.pt')
            if not os.path.exists(save_path):
                quant_param_ckpt = torch.load(os.path.join(args.log, args.quant_param_ckpt), weights_only=True, map_location='cuda')
                model.load_quant_param_dict(quant_param_ckpt)
                model.quantize_and_save_weight(save_path=save_path)
            else:
                logger.info('int_weight.pth exists, loading from the local file...')
            model.hardware_forward_refactor(load_path=save_path)

    else:  # use the algorithm simulation
        quant_param_ckpt = torch.load(os.path.join(args.log, args.quant_param_ckpt), weights_only=True)
        model.load_quant_param_dict(quant_param_ckpt)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [217, 363, 347, 574, 188, 99, 47, 379]
    #class_labels = [47]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=torch.float16)
    y = torch.tensor(class_labels, device=device)  # type: long, not half. 

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0).contiguous()
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0).contiguous()
    model_kwargs = dict(y=y)
    t = torch.tensor([1] * z.shape[0], device=device, dtype=torch.float16).contiguous()

    if args.profile:
        # init the FP model also
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        fp_model=DiT(
         input_size=latent_size,
         patch_size=2, 
         in_channels=4,
         hidden_size=1152, 
         depth=28,
         num_heads=16, 
         num_classes=args.num_classes,
         ).to(device)
        fp_model.half()

        state_dict = find_model(ckpt_path)
        fp_model.load_state_dict(state_dict, strict=False)
        fp_model.eval()  # important!

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # record the FP  model inference time.
        num_iter = 100
        for i in range(num_iter):
            if i != 0:  # skip the 1st iter
                start.record()
            _ = fp_model(z,t,y)
        end.record()
        torch.cuda.synchronize()
        avg_time = start.elapsed_time(end) / (num_iter-1)
        print(f"FP inference {avg_time:.2f}ms")

        # record the quantzied model inference time.
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(num_iter):
            if i != 0:  # skip the 1st iter
                start.record()
            _ = model(z,t,y)
        end.record()
        torch.cuda.synchronize()
        avg_time = start.elapsed_time(end) / (num_iter-1)
        print(f"quantized inference {avg_time:.2f}ms")

    model.set_init_done()
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # save the images in local folder
    save_image(samples, os.path.join(args.log, 'quantized_img.png'), nrow=4, normalize=True, value_range=(-1, 1))

    if quant_config.get('smooth_quant',None) is not None: # record the alpha also, for sweep alpha
        if not os.path.exists('./imgs'):
            os.makedirs('./imgs')
        save_image(samples, "./imgs/sample_{:.4f}.png".format(quant_config.smooth_quant.alpha), nrow=4, normalize=True, value_range=(-1, 1))

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
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
