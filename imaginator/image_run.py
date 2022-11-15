"""
Stable Diffusion img2img for Kids' Artwork

Brooke Hamilton
brookehamilton@gmail.com

The ImageRun class takes in an inference config and a starting image and produces a resulting image
"""
from . import configs
from diffusers import StableDiffusionImg2ImgPipeline
from .pipeline_utils import create_pipeline, resize_image
import torch
import PIL
from PIL import Image
import numpy as np
import json

class ImageRun():

    def __init__(self,
            run_config: configs.InferenceConfig,
            pipeline: StableDiffusionImg2ImgPipeline = None,
            resize_pixels=None,
            init_image_path=None,
            safety_checker=False):

        # Get run parameters from config
        self.run_config = run_config
        if self.run_config.seed is None:
            self.run_config.seed = np.random.randint(1, 1000000)

        # Set up pipeline
        self.pipeline = pipeline
        if self.pipeline is None:
            self.pipeline = create_pipeline()
        if not safety_checker:
            self.turn_off_safety_checker()

        # Initial image
        self.init_image = None
        self.resize_pixels = resize_pixels
        self.init_image_path = init_image_path
        if self.init_image_path is not None:
            self.load_init_image(init_image_path=self.init_image_path)

        # Created image
        self.image = None

    def load_init_image(self, init_image_path: str):
        """
        Load the initial image from path
        """
        self.init_image_path = init_image_path
        init_image = Image.open(self.init_image_path).convert("RGB")
        if self.resize_pixels is not None:
            init_image = resize_image(init_image, desired_max_dimension=self.resize_pixels)
            #(f'Resized init image to {self.resize_pixels} pixels on longest side')
        self.init_image = init_image

    def load_init_image_from_PIL(self, init_image: PIL.Image):
        """Load an initial image that is already a PIL Image object, for example from a Gradio input"""
        self.init_image = init_image

    def turn_off_safety_checker(self):
        """
        Turn off the default NSFW safety checker from Huggingface. I have found that it has a high
        rate of false positives.
        """
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def create_image(self):
        """
        Generate an image using the pipeline
        """
        if self.init_image is None:
            raise RuntimeError('No initial image selected. Please load initial image with load_init_image() and try again.')

        generator = torch.manual_seed(self.run_config.seed) if not torch.cuda.is_available() else torch.cuda.manual_seed(self.run_config.seed)
        images = self.pipeline(prompt=self.run_config.prompt,
                    init_image=self.init_image,
                    strength=self.run_config.strength,
                    guidance_scale=self.run_config.guidance_scale,
                    negative_prompt=self.run_config.negative_prompt,
                    num_inference_steps = self.run_config.num_inference_steps,
                    generator=generator
                    ).images
        self.image = images[0]

    def save_image(self, image_out_path: str):
        """Save the image to disk"""
        self.image.save(image_out_path)
        print(f'Image saved to {image_out_path}')

    def save_run_parameters(self, out_path='run_parameters.json'):
        """Save the parameters used for this run as a JSON file"""

        run_parameters = {}
        run_parameters['init_image_path'] = self.init_image_path
        run_parameters['seed'] = self.run_config.seed
        run_parameters['prompt'] = self.run_config.prompt
        run_parameters['negative_prompt'] = self.run_config.negative_prompt
        run_parameters['strength'] = self.run_config.strength
        run_parameters['guidance_scale'] = self.run_config.guidance_scale
        run_parameters['num_inference_steps'] = self.run_config.num_inference_steps

        with open(out_path, 'w') as f:
            json.dump(run_parameters, f)
        print(f'JSON with run parameters saved to {out_path}')
