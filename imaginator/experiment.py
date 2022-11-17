"""
Experiment Runner

Brooke Hamilton
brookehamilton@gmail.com

This script contains a SeedExperiment class that helps the user iterate on a specific piece of art more effectively.
"""
from PIL import Image
import copy
from diffusers import StableDiffusionImg2ImgPipeline
from imaginator.pipeline_utils import create_pipeline
from imaginator.configs import InferenceConfig
from imaginator.image_run import ImageRun
from imaginator.image_utils import resize_image, image_grid

class SeedExperiment():

    def __init__(self,
            run_config: InferenceConfig,
            pipeline: StableDiffusionImg2ImgPipeline = None,
            resize_pixels=None,
            init_image_path=None,
            safety_checker=False,
            num_runs=9):

        # Get run parameters from config
        self.run_config = run_config
        self.run_config.seed = None   # make sure seed is None for Seed Experiment -- will be set within ImageRun objects
        self.safety_checker = safety_checker

        # Experiment
        self.num_runs = num_runs

        # Set up pipeline
        self.pipeline = pipeline
        if self.pipeline is None:
            self.pipeline = create_pipeline()

        # Initial image
        self.resize_pixels = resize_pixels
        self.init_image_path = init_image_path

        # Experimental runs
        self.runs = None

    def initialize_runs(self, num_inference_steps=10):

        # Initialize a list containing ImageRun objects
        self.runs = []

        for i in range(self.num_runs):
            this_run_config = copy.deepcopy(self.run_config)
            this_run_config.num_inference_steps = num_inference_steps
            run = ImageRun(run_config = this_run_config,
            pipeline=self.pipeline,
            resize_pixels=self.resize_pixels,
            init_image_path=self.init_image_path,
            safety_checker=self.safety_checker)
            self.runs.append(run)

    def downsize_run_images(self, downsize_pixels=500):
        """
        For each experimental run, downsize the initial image to make the run go faster.

        N.B. It is not recommended to reduce the size smaller than ~500px
        """
        for i in self.runs:
            i.init_image = resize_image(i.init_image, desired_max_dimension=downsize_pixels)

    def run_all(self, num_inference_steps=10):
        """
        Kick off image creation in each of the experimental runs
        """
        if self.runs is None:
            self.initialize_runs(num_inference_steps=num_inference_steps)

        for i in self.runs:
            i.create_image()

    def show_images(self):
        """
        Display the created images in a grid.
        Note: this assumes you're making 9 images. Grid will fail otherwise.
        """
        images = [i.image for i in self.runs]

        return image_grid(images, 3, 3)

    def scale_up_run(self, run_index: int, num_inference_steps: int = 50):
        """
        For a given experimental run, re-run the image creation with a larger
        number of inference steps to create the final image
        """

        run = self.runs[run_index]
        run.run_config.num_inference_steps = num_inference_steps
        run.create_image()
