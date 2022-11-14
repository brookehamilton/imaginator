"""
Stable Diffusion img2img for Kids' Artwork

Brooke Hamilton
brookehamilton@gmail.com
"""
import click
import os
from imaginator.pipeline_utils import create_pipeline, run_image_creation
from imaginator.configs import InferenceConfig
from imaginator.image_run import ImageRun
import datetime

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


def run(init_image_path, prompt, negative_prompt, output_dir, output_filename, num_inference_steps, seed, strength, guidance_scale, safety_checker):
    """
    Main function to create an image from with command line with a starting image and a prompt
    """
    # Set up inference config
    config_dict = {'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'seed': seed
            }
    inference_config = InferenceConfig(**config_dict)

    # Create pipeline
    pipe = create_pipeline()

    # Run image creation
    Run = ImageRun(run_config = inference_config, pipeline=pipe, resize_pixels=700,
               init_image_path=init_image_path, safety_checker=safety_checker)
    Run.create_image()

    # Save image and run parameters
    # If no filename or directory specified, create them
    if output_dir is None:
        output_dir = datetime.datetime.now().strftime('creations/%Y_%m_%d__%H_%M_%s/')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    if output_filename is None:
        output_filename = 'output.jpg'
    image_out_path = os.path.join(output_dir, output_filename)
    Run.save_image(image_out_path)
    Run.save_run_parameters(out_path = os.path.join(output_dir, 'run_parameters.json'))

if __name__ == '__main__':
    run()
