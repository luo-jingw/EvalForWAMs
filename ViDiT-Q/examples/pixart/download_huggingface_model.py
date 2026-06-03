# the code for generating the `local` diffusers pipeline
# 1st load the pipeline using automatic download, then use `save_pretrained` to generate local file
from diffusers import PixArtSigmaPipeline

model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

cache_dir = "/share/public/diffusion_quant/huggingface/hub/"

save_path = "./pretrained_models/"  # change accordingly

pipe = PixArtSigmaPipeline.from_pretrained(model_id, cache_dir = cache_dir)
pipe.save_pretrained(save_path)

