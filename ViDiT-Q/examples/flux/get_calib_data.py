import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging

from diffusers import FluxPipeline
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from omegaconf import OmegaConf
import torch.nn as nn

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
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    ckpt_path = args.ckpt if args.ckpt is not None else "./pretrained_models/"
    pipe = FluxPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    # INFO: if memory intense
    pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()
    
    model = pipe.transformer
    quant_config = OmegaConf.load(args.quant_config)
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

    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images

    save_d = {}
    for k,v in hook_d.items():
        save_d[k] = torch.stack(v.outputs, dim=0) # [N_timestep*B, C]  -> [C]
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()

    torch.save(save_d, os.path.join(args.log, quant_config.calib_data.save_path))
    logger.info(f'saved calib data in {os.path.join(args.log, quant_config.calib_data.save_path)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
