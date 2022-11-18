"""
Experiment Runner

Brooke Hamilton
brookehamilton@gmail.com

This script contains classes that helps the user iterate on a specific piece of art more effectively.

Experiment: base class that sets up a number of ImageRun instances

SeedExeriment: class that generates a group of ImageRuns, each with a different seed, run for a short number of steps. Choose the best
resulting image and scale it up with more steps.

"""
from PIL import Image
import copy
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from imaginator.pipeline_utils import create_pipeline
from imaginator.configs import InferenceConfig
from imaginator.image_run import ImageRun
from imaginator.image_utils import resize_image, image_grid
from typing import List


class Experiment():

    def __init__(self,
            run_configs: List[InferenceConfig] = None,
            pipeline: StableDiffusionImg2ImgPipeline = None,
            resize_pixels: int = None,
            init_image_path: str = None,
            safety_checker: bool = False):

        # Set up pipeline
        self.pipeline = pipeline
        if self.pipeline is None:
            self.pipeline = create_pipeline()
        self.safety_checker = safety_checker

        # Starting image
        self.init_image = None
        self.init_image_path = init_image_path
        self.resize_pixels = resize_pixels

        # Experimental Runs
        self.run_configs = run_configs
        self.runs = None

        # Final Run
        self.final_run = None

        # Initialize and run everything
        self.run_all()

    def initialize_runs(self):
        # Initialize a list containing ImageRun objects
        self.runs = []

        for i in self.run_configs:
            run = ImageRun(
                run_config = i,
                pipeline=self.pipeline,
                resize_pixels=self.resize_pixels,
                init_image_path=self.init_image_path,
                safety_checker=self.safety_checker
                )
            self.runs.append(run)

    def run_all(self):
        """
        Kick off image creation in each of the experimental runs
        """
        if self.runs is None:
            self.initialize_runs()

        for i in self.runs:
            i.create_image()

    def show_images(self):
        """
        Display the created images in a grid.
        Note: this assumes you're making 9 images. Grid will fail otherwise.
        """
        images = [i.image for i in self.runs]

        return image_grid(images, 3, 3)




class SeedExperiment(Experiment):
    def __init__(self,
            run_config: InferenceConfig,
            num_runs: int = 9,
            pipeline: StableDiffusionImg2ImgPipeline = None,
            resize_pixels = None,
            init_image_path = None,
            safety_checker = False):

        self.run_config = run_config
        self.run_config.seed = None     # Make sure no seed is set
        self.num_runs = num_runs
        self.final_image = None

        experiment_run_configs = self.create_run_configs()
        super().__init__(run_configs=experiment_run_configs,
                        pipeline=pipeline,
                        resize_pixels=resize_pixels,
                        init_image_path=init_image_path,
                        safety_checker=safety_checker
                        )

    def create_run_configs(self):
        experiment_run_configs = []
        for i in range(self.num_runs):
            this_run_config = copy.deepcopy(self.run_config)
            this_run_config.num_inference_steps = 2
            this_run_config.seed = np.random.randint(1, 1000000)
            experiment_run_configs.append(this_run_config)
        return experiment_run_configs

    def scale_up_run(self, run_index: int, num_inference_steps: int = 50):
        """
        For a given experimental run, re-run the image creation with a larger
        number of inference steps to create the final image
        """

        run = self.runs[run_index]
        run.run_config.num_inference_steps = num_inference_steps
        run.create_image()

        self.final_image = run.image
        return self.final_image




'''class SeedExperiment_old():

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

        # Final image created by scaling up the chosen best candidate
        self.final_image = None

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

        self.final_image = run.image
        return self.final_image
'''
