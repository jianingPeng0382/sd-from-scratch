import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
import random
import warnings
import os

warnings.filterwarnings('ignore')

DEVICE = "cpu"

ALLOW_CUDA  = True
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda:0"
# elif {torch.has_mps or torch.balanced_memory_enabled} and ALLOW_MPS:
#     DEVICE = "cuda"
print("Using device: ", DEVICE)

tokenizer = CLIPTokenizer("/opt/data/private/pjn/sd-from-scratch/data/tokenizer_vocab.json", merges_file="/opt/data/private/pjn/sd-from-scratch/data/tokenizer_merges.txt")
model_file = "/opt/data/private/pjn/sd-from-scratch/data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Set the prompt and unconditional prompt
prompt_for_inversion = "a cat in a field of grass"
prompt_for_reconstruct_or_edit = "a cat in a field of flowers"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7.5
seed = None

# Set the sampler
sampler = "ddim"

# if use ddpm, set the number of inference steps
if sampler == "ddpm":
    num_inference_steps = 50

# if use ddim, set the number of inference steps and sample_steps
sample_steps = None
if sampler == "ddim":
    num_inference_steps = 50
    sample_steps = 10
# true_num_inference_steps_of_ddim = num_inference_steps // sample_steps

# Inversion for Image
is_inversion = True
is_input_image = True
is_invert_for_latents = False
    

# Set the input image path and strength for the image to image generationor 
# or image to latents inversion
input_image = None
strength = 0.9
cfg_scale_inversion = None
if is_input_image:
    image_path = "/opt/data/private/pjn/sd-from-scratch/images/output_3.jpg"
    input_image = Image.open(image_path)
    cfg_scale_inversion = 1


# if use ddim and inversion, set the inverted latents
# Generate a image from the inverted latents
inverted_latents = None



output_latents = pipeline.generate(
    prompt=prompt_for_inversion,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    sample_steps=sample_steps,
    is_inversion=is_inversion,
    cfg_scale_inversion=cfg_scale_inversion,
    inverted_latents=inverted_latents,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cuda:0",
    tokenizer=tokenizer
    )

   
# Inverted Latents to Image
input_image = None
cfg_scale_inversion = None

is_inversion = False
is_input_image = False
is_invert_for_latents = True

if is_invert_for_latents and sampler == "ddim":
    inverted_latents = output_latents
    inverted_latents.to(DEVICE)

output_image = pipeline.generate(
    prompt=prompt_for_reconstruct_or_edit,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    sample_steps=sample_steps,
    is_inversion=is_inversion,
    cfg_scale_inversion=cfg_scale_inversion,
    inverted_latents=inverted_latents,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cuda:0",
    tokenizer=tokenizer
    )


inverted_latents_name = os.path.basename(image_path)
inverted_latents_path = f"/opt/data/private/pjn/sd-from-scratch/inverted_latents/latents_of_{inverted_latents_name}.pt"
print(f"Inverted latents of: {inverted_latents_name}")
torch.save(output_latents, inverted_latents_path)


reconsructed_or_edited_latents_name = os.path.basename(inverted_latents_path)
output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/reconstructed(edited)_img_from_{reconsructed_or_edited_latents_name}.jpg"
print(f"reconstructed(edited)_img_from_latents_{reconsructed_or_edited_latents_name}")
Image.fromarray(output_image).save(output_image_path)
