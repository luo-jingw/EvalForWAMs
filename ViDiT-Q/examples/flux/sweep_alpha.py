import subprocess
from omegaconf import OmegaConf
import numpy as np
import os
import shutil

def modify_config(config_path, new_alpha):
    config = OmegaConf.load(config_path)
    if 'smooth_quant' in config and 'alpha' in config.smooth_quant:
        config.smooth_quant.alpha = new_alpha
    elif 'viditq' in config and 'alpha' in config.viditq:
        config.viditq.alpha = new_alpha
    else:
        raise ValueError("The configuration file does not contain 'smooth_quant.alpha'")
    OmegaConf.save(config, config_path)

def execute_shell_script(script_path):
    subprocess.run(["bash", script_path])

if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    log_path = './logs/sweep_alpha'
    alphas = np.arange(0.1, 1.0, 0.1)

    for alpha in alphas:
        print(f"Setting smooth_quant.alpha to {alpha}")
        modify_config(config_path, float(alpha))
        script_path = "./main.sh"
        execute_shell_script(script_path)
        if os.path.exists(os.path.join(log_path, 'generated_images_{:.4f}'.format(alpha))):
            shutil.rmtree(os.path.join(log_path, 'generated_images'))
            continue
        os.rename(os.path.join(log_path,'generated_images'), os.path.join(log_path, 'generated_images_{:.4f}'.format(alpha)))
