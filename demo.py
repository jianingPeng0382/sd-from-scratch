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
prompt = "a cat with the white background"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7.5

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


# Set the function of model
# Text to Image: False, False, False
# Image to Image: False, True, False
# Inversion for Image: True, True, False
# Inverted Latents to Image: False, False, True

# Inversion for Image
is_inversion = True
is_input_image = True
is_invert_for_latents = False
    
# Inverted Latents to Image
# is_inversion = False
# is_input_image = False
# is_invert_for_latents = True

# Set the input image path and strength for the image to image generationor 
# or image to latents inversion
input_image = None
strength = 0.9
cfg_scale_inversion = None
if is_input_image:
    image_path = "/opt/data/private/pjn/sd-from-scratch/images/cat.jpg"
    input_image = Image.open(image_path)
    cfg_scale_inversion = 1


# if use ddim and inversion, set the inverted latents
# Generate a image from the inverted latents
inverted_latents = None
if is_invert_for_latents and sampler == "ddim":
    inverted_latents_path = "/opt/data/private/pjn/sd-from-scratch/inverted_latents/latents_of_cat.jpg.pt"
    inverted_latents = torch.load(inverted_latents_path)
    inverted_latents.to(DEVICE)

# Generate multiple images with different seeds
def generate_random_seeds(num_seeds):
    seeds = []
    for _ in range(num_seeds):
        seed = random.randint(0, 9999)
        seeds.append(seed)
    return seeds

num_seeds = 1
seeds = generate_random_seeds(num_seeds)

for seed in seeds:
        output_image = pipeline.generate(
        prompt=prompt,
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
        
if is_input_image:
    inverted_latents_name = os.path.basename(image_path)
if is_invert_for_latents:
    reconsructed_or_edited_latents_name = os.path.basename(inverted_latents_path)

if is_inversion and (not is_invert_for_latents) and is_input_image:
    inverted_latents_path = f"/opt/data/private/pjn/sd-from-scratch/inverted_latents/latents_of_{inverted_latents_name}.pt"
    print(f"Inverted latents of: {inverted_latents_name}")
    torch.save(output_image, inverted_latents_path)

elif (not is_inversion) and (not is_invert_for_latents) and (not is_input_image):
    output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/output_seed_{seed}.jpg"
    print(f"T2I image of Seed: {seed}")
    Image.fromarray(output_image).save(output_image_path)

elif (not is_inversion) and is_invert_for_latents and (not is_input_image):
    output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/reconstructed(edited)_img_from_{reconsructed_or_edited_latents_name}.jpg"
    print(f"reconstructed(edited)_img_from_latents_{reconsructed_or_edited_latents_name}")
    Image.fromarray(output_image).save(output_image_path)

elif (not is_inversion) and (not is_invert_for_latents) and is_input_image:
    output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/output_seed_{seed}.jpg"
    print(f" I2I image of Noise Strength: {strength}")
    Image.fromarray(output_image).save(output_image_path)

else:
    print("Error: Invalid settings for is_inversion, is_input_image and is_invert_for_latents")
# inverted_latents_name = os.path.basename(inverted_latents_path)
#     output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/output_{inverted_latents_name}.jpg"

# # Generate images with different numbers of inference steps

# seed = 2574

# step_length = 10
# generate_times = 10
# inference_steps_list = [step_length * i for i in range(1, generate_times + 1)]


# for i, num_inference_steps in enumerate(inference_steps_list):
#     output_image = pipeline.generate(
#         prompt=prompt,
#         uncond_prompt=uncond_prompt,
#         input_image=input_image,
#         strength=strength,
#         do_cfg=do_cfg,
#         cfg_scale=cfg_scale,
#         sampler_name=sampler,
#         n_inference_steps=num_inference_steps,
#         seed=seed,
#         models=models,
#         device=DEVICE,
#         idle_device="cuda:0",
#         tokenizer=tokenizer
#     )
#     output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/output_seed_{seed}_steps_{num_inference_steps}.jpg"
#     Image.fromarray(output_image).save(output_image_path)

# # Generate images with different strengths
# strengths = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# seed = 2574

# for strength in strengths:
#     output_image = pipeline.generate(
#         prompt=prompt,
#         uncond_prompt=uncond_prompt,
#         input_image=input_image,
#         strength=strength,
#         do_cfg=do_cfg,
#         cfg_scale=cfg_scale,
#         sampler_name=sampler,
#         n_inference_steps=num_inference_steps,
#         seed=seed,
#         models=models,
#         device=DEVICE,
#         idle_device="cuda:0",
#         tokenizer=tokenizer
#     )
#     output_image_path = f"/opt/data/private/pjn/sd-from-scratch/images/output_seed_{seed}_strength_{strength}.jpg"
#     Image.fromarray(output_image).save(output_image_path)
