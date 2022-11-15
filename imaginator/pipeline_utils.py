"""
Stable Diffusion - Pipeline Utility Functions

Brooke Hamilton
brookehamilton@gmail.com

This script has functions that are wrappers around the huggingface diffusion library
"""

import torch
from torch import autocast
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import getpass
import datetime
import json
import os
import PIL
from imaginator.image_utils import resize_image


def create_pipeline():
    """
    Create an img2img pipeline from pretrained Stable Diffusion model
    """

    # Prompt the user for their HuggingFace token
    if "HUGGINGFACE_ACCESS_TOKEN" in os.environ:
        access_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']
    else:
        access_token = getpass.getpass(prompt='HuggingFace access token:')

    # Set up the img2img pipeline from pretrained Stable Diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #device = "cuda"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        #torch_dtype=torch.float16,
        use_auth_token=access_token
    )
    pipe = pipe.to(device)
    print(f'Created Img2Img Pipeline using: {device}')
    return pipe

def run_pipeline(pipe,
                prompt,
                init_image,
                strength=0.75,
                guidance_scale=7.5,
                negative_prompt=None,
                num_inference_steps=15,
                seed=None):
    """
    Generate an image using the pipeline
    """
    #init_image = init_image.resize((768, 512))

    generator = torch.manual_seed(seed) if not torch.cuda.is_available() else torch.cuda.manual_seed(seed)
    image = pipe(prompt=prompt,
                init_image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_inference_steps = num_inference_steps,
                generator=generator
                ).images
    return image

def create_image(
            init_image_path,
            prompt,
            pipe=None,
            negative_prompt=None,
            num_inference_steps=15,
            seed=None,
            strength=0.75,
            guidance_scale=7.5,
            safety_checker=False,
            resize=True):
    """
    Create a single image, given a prompt and a starting image

    pipe:
        the stable diffusion pipeline object, created with create_pipeline()
    init_image_path:
        local path to image you want to start with (i.e., kid's artwork)
    prompt:
        a text description of what you want the final image to look like
    negative_prompt:
        a text description (str or list) of what you don't want the output to look like
    seed:
        random number that will generate the same image each time
    strength:
        How much to transform the reference init_image. Range (0,1), where 0 should
        look very similar to init image, and 1 would essentially ignore the init image
    guidance_scale:
        forces the generation to better match the text prompt, potentially at
        the cost of image quality or diversity. Values between 7 and 8.5 are usually good choices
    safety_checker:
        if True, filter out potentially NSFW images from results (shows black box)
        Note: I turned this off by default because it seems to have a very high false positive rate --
        it was flagging ~50% of my benign prompts as NSFW

    For more details about available pipeline parameters, see:
    https://huggingface.co/docs/diffusers/v0.5.1/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionImg2ImgPipeline
    """
    if pipe is None:
        pipe = create_pipeline()

    init_image = Image.open(init_image_path).convert("RGB")

    if resize:
        init_image = resize_image(init_image)

    if seed is None:
        seed = np.random.randint(1, 1000000)

    if not safety_checker:
        pipe.safety_checker = lambda images, clip_input: (images, False)

    images = run_pipeline(pipe=pipe,
                prompt=prompt,
                init_image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_inference_steps = num_inference_steps,
                seed=seed)

    # Create a dictionary to return the image and all parameters we may want to keep track of
    # Why? Because it is really common to generate an image and then forget what prompt you used,
    # or the seed, and you may want to generate it again
    return_dict = {}
    return_dict['image'] = images[0]
    return_dict['init_image_path'] = init_image_path
    return_dict['seed'] = seed
    return_dict['prompt'] = prompt
    return_dict['negative_prompt'] = negative_prompt
    return_dict['strength'] = strength
    return_dict['guidance_scale'] = guidance_scale

    return return_dict

def run_image_creation(
            init_image_path,
            prompt,
            output_dir=None,
            output_filename=None,
            pipe=None,
            negative_prompt=None,
            num_inference_steps=15,
            seed=None,
            strength=0.75,
            guidance_scale=7.5,
            safety_checker=False):
    """
    This is a wrapper around create_image(), allowing the user to save the outputs to a specified directory

    If no output directory is specified, one will be created in the creations directory, using the current timestamp,
    for example: 'creations/2022_10_21__10_19_1666365581/'
    """
    # Create the image
    results = create_image(pipe=pipe,
                init_image_path=init_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
                strength=strength,
                guidance_scale=guidance_scale,
                safety_checker=safety_checker)

    # Write image to disk
    if output_dir is None:
        output_dir = datetime.datetime.now().strftime('creations/%Y_%m_%d__%H_%M_%s/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if output_filename is None:
        output_filename = 'output.jpg'
    out_path = os.path.join(output_dir, output_filename)
    results['image'].save(out_path)
    print(f'Image saved to {out_path}')

    # Write parameters to disk as JSON
    json_to_write = {k: results[k] for k in results if k != 'image'}
    out_json_path = os.path.join(output_dir, 'run_parameters.json')
    with open(out_json_path, 'w') as f:
        json.dump(json_to_write, f)
    print(f'JSON with run parameters saved to {out_json_path}')

    return results
