"""
Stable Diffusion img2img for Kids' Artwork

Brooke Hamilton
brookehamilton@gmail.com
"""

import click
#from utils.pipeline_utils import *

@click.command()
@click.option('--verbose', default=True, help='Print out run information and pop up window with resulting image(s)')
@click.option('--init_image_path', required=True, help='Local path to image file you want to start from (e.g. kid\'s artwork)')
@click.option('--prompt', required=True, help='A text description of what you want the final image to look like')
@click.option('--negative_prompt', help='A text description (str or list) of what you don\'t want the output to look like')
@click.option('--num_inference_steps', help='A text description (str or list) of what you don\'t want the output to look like')
@click.option('--seed', help='Random number that will generate the same image each time')
@click.option('--strength', help='How much to transform the reference init_image. Range (0,1)')
@click.option('--guidance_scale', help='Forces the generation to better match the text prompt. Range (7,8) recommended.')
@click.option('--safety_checker', help='If True, return black box for potentially NSFW images')



#@click.option('--count', default=1, help='Number of greetings.')

def run(verbose, init_image_path, prompt, negative_prompt, num_inference_steps, seed, strength, guidance_scale, safety_checker):
    """Simple program that greets NAME for a total of COUNT times."""
    #create_pipeline()
    #create_image()
    print(f'init_image_path: {init_image_path}')
    print(f'prompt: {prompt}')
    print(f'negative_prompt: {negative_prompt}')

if __name__ == '__main__':
    run()