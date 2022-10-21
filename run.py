"""
Stable Diffusion img2img for Kids' Artwork

Brooke Hamilton
brookehamilton@gmail.com
"""
import click
from utils.pipeline_utils import *

@click.command()
@click.option('--init_image_path', required=True, help='Local path to image file you want to start from (e.g. kid\'s artwork)')
@click.option('--prompt', required=True, help='A text description of what you want the final image to look like')
@click.option('--negative_prompt', help='A text description (str or list) of what you don\'t want the output to look like')
@click.option('--output_dir', help='Directory where you want generated image to be written to')
@click.option('--output_filename', help='Filename where you want generated image to be written to')
@click.option('--num_inference_steps', default=25, help='A text description (str or list) of what you don\'t want the output to look like')
@click.option('--seed', help='Random number that will generate the same image each time')
@click.option('--strength', default=0.75, help='How much to transform the reference init_image. Range (0,1)')
@click.option('--guidance_scale', default=7.5, help='Forces the generation to better match the text prompt. Range (7,8) recommended.')
@click.option('--safety_checker', default=False, help='If True, return black box for potentially NSFW images')

def run(init_image_path, 
        prompt, 
        output_dir, 
        output_filename,
        negative_prompt, 
        num_inference_steps, 
        seed, 
        strength, 
        guidance_scale, 
        safety_checker):
    """
    Main function to create an image from with command line with a starting image and a prompt
    """
    pipe = create_pipeline()
    print('Generating image with these parameters:')
    print(f'init_image_path: {init_image_path}')
    print(f'prompt: {prompt}')
    print(f'negative_prompt: {negative_prompt}')
    results = run_image_creation(init_image_path=init_image_path, 
            prompt=prompt, 
            output_dir=output_dir,
            output_filename=output_filename,
            pipe=pipe,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed, 
            strength=strength, 
            guidance_scale=guidance_scale,
            safety_checker=safety_checker)


if __name__ == '__main__':
    run()