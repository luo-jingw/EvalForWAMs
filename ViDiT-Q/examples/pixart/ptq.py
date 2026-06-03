import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging

from diffusers import PixArtSigmaPipeline
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

from models.customize_pipeline_pixart_sigma import CustomizePixArtSigmaPipeline
from models.customize_pipeline_pixart_transformer_2d import CustomizePixArtTransformer2DModel
# DIRTY: apply monkey patch, since the from_pretrained() method is hard to hack
diffusers.models.PixArtTransformer2DModel = CustomizePixArtTransformer2DModel
diffusers.PixArtSigmaPipeline = CustomizePixArtSigmaPipeline
from diffusers import PixArtSigmaPipeline
from omegaconf import OmegaConf

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
    
    # INFO: backup a few files
    import shutil
    shutil.copy('./configs/config.yaml', args.log)
    if os.path.exists(os.path.join(args.log,'models')):
        shutil.rmtree(os.path.join(args.log,'models'))
    shutil.copytree('./models', os.path.join(args.log,'models'))

    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # ---- assign quant configs ------
    quant_config = OmegaConf.load(args.quant_config)
    print(quant_config)
    pipe.convert_quant(quant_config)
    model = pipe.transformer

    '''
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''
    def init_sq_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, SQQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.update_quantized_weight_scaled()

    def init_rotation_matrix_(module, full_name):
        from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
        assert isinstance(module, QuarotQuantizedLinear)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated()
    
    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, ViDiTQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    '''
    INFO: the smooth_quant quantization.
    load act channel mask from the calib data
    '''
    if quant_config.get("smooth_quant",None) is not None:
        # INFO: the SQQuantizedLayer are initialized with the quant_layer_refactor_ in quant_dit.py
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
        
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        # get the rotation matrix, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=QuarotQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_matrix_,
                            full_name='',
                            **kwargs
                            )
    '''
    INFO: combining both
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(os.path.join(args.log, quant_config.calib_data.save_path), weights_only=True)  # default wtih 
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )
        
    model.set_init_done()
    model.save_quant_param_dict()
    torch.save(pipe.transformer.quant_param_dict, os.path.join(args.log, 'quant_params.pth'))
    logger.info(f'saved quant params into {args.log}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
