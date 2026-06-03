import torch
from diffusers import FluxPipeline

cache_dir = "/share/xieruiqi-local/cache/flux/"
save_path = "/share/xieruiqi-local/models/flux/" # change accordingly

pipe = FluxPipeline.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
# pipe.save_pretrained(save_path)

# Test with generate a photo.
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")