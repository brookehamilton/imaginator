"""
Stable Diffusion - DreamBooth

Brooke Hamilton
brookehamilton@gmail.com

Tools for training a DreamBooth model. Adapted from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
"""
import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataclasses import dataclass
import getpass
import datetime



@dataclass
class DreamBoothConfig():
    """
    This class holds all of the config values for the overall DreamBooth process
    """
    # saved model
    model_out_dir: str      # directory to save the model in

    # the pretrained model we want to use
    model_id: str               # e.g. 'runwayml/stable-diffusion-v1-5
    revision: str               # usually None

    # the specific images you want to teach it to replicate
    instance_prompt: str        # e.g. 'a photograph of a vctrnxmas Victorian Christmas card painting'
    instance_data_dir: str

    # prior preservation -- additional images of the generic class
    do_prior_preservation: bool
    generate_class_images: bool     # whether to generate fresh class images w/ model (otherwise will assume images are already in directory specified)
    class_prompt: str               # e.g. 'a photograph of a Victorian Christmas card painting'
    class_data_dir: str
    num_class_images: int
    sample_batch_size: int

    # image settings
    resolution: int                     # = 512
    center_crop: bool                   # = False

    # training settings
    seed: int                           # = None
    train_text_encoder: bool            # False
    gradient_checkpointing: bool        # False
    learning_rate: int                  # 5e-6,
    use_8bit_adam: bool                 # False,
    adam_beta1: float                   # 0.9
    adam_beta2: float                   # 0.999
    adam_weight_decay: float            # 1e-2
    adam_epsilon: float                 # 1e-08
    max_grad_norm: float                # 1.0
    train_batch_size: int               # 4
    num_train_epochs: int               # 1?
    max_train_steps: int                # 800?
    gradient_accumulation_steps: int    # 1
    lr_scheduler: str                   # "constant"
    lr_warmup_steps: int                # 500
    save_steps: int
    prior_loss_weight: float

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_dir,
        instance_prompt,
        tokenizer,
        class_data_dir=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_dir = Path(instance_data_dir)
        if not self.instance_data_dir.exists():
            raise ValueError("Instance images directory doesn't exist.")

        self.instance_images_path = list(Path(instance_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_dir is not None:
            print('class_data_dir is not None')
            self.class_data_dir = Path(class_data_dir)
            self.class_data_dir.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_dir.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_dir = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_dir:
            print('triggered at if self.class_data_dir')
            print('self.num_class_images: ', self.num_class_images)
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

'''class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and then tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_dir,
        instance_prompt,
        tokenizer,
        class_data_dir=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_dir = Path(instance_data_dir)
        if not self.instance_data_dir.exists():
            raise ValueError("Instance images directory doesn't exist.")

        self.instance_images_path = list(Path(instance_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_dir is not None:
            class_data_dir = Path(class_data_dir)
            class_data_dir.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(class_data_dir.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_dir = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        """
        Get a specific example by the index number

        N.B. If the number of instance images and class images is not equal, examples will recycle images as needed
        """
        print(f'getting example {index}')
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_dir:
            print('self.class_images_path:', self.class_images_path)
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example'''


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example



class Collater():
    def __init__(self, do_prior_preservation, tokenizer):
        self.do_prior_preservation = do_prior_preservation
        self.tokenizer = tokenizer
    def __call__(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if self.do_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch


class DreamBoothRunner():
    """
    This class coordinates a dream booth training job
    """
    def __init__(self, DreamBoothConfig: DreamBoothConfig):
        print('='*50)
        print('Initialized DreamBooth model training job')
        print('='*50)
        self.config = DreamBoothConfig
        if self.config.seed is not None:
            print(f'Setting seed to {self.config.seed}')
            set_seed(self.config.seed)

        # Make sure model_out_dir exists
        os.makedirs(self.config.model_out_dir, exist_ok=True)

        # Log in to Hugging Face
        print('Authenticating to HuggingFace')
        if "HUGGINGFACE_ACCESS_TOKEN" in os.environ:
            self.auth_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']
        else:
            self.auth_token = getpass.getpass(prompt='HuggingFace access token:')

        print('Starting Accelerator')
        self.accelerator = Accelerator(gradient_accumulation_steps=1,
            mixed_precision="fp16" if torch.cuda.is_available() else "no", #["no", "fp16", "bf16"]
            log_with="tensorboard",
            logging_dir='logs')

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
            revision=self.config.revision,
            use_auth_token=self.auth_token
        )

        # Load models and create wrapper for stable diffusion
        print('Getting CLIPTextModel text encoder')
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            revision=self.config.revision,
            use_auth_token=self.auth_token
        )
        if not self.config.train_text_encoder:
            self.text_encoder.requires_grad_(False)
        if self.config.train_text_encoder:
            if self.config.gradient_checkpointing:
                self.text_encoder.gradient_checkpointing_enable()

        print('Getting AutoencoderKL vae')
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            revision=self.config.revision,
            use_auth_token=self.auth_token
        )
        self.vae.requires_grad_(False)

        print('Gettting UNet2DConditionModel unet')
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_id,
            subfolder="unet",
            revision=self.config.revision,
            use_auth_token=self.auth_token
        )
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        print('Getting DDPMScheduler noise scheduler')
        self.noise_scheduler = DDPMScheduler.from_config(
            self.config.model_id,
            subfolder="scheduler",
            use_auth_token=self.auth_token)

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        print('Setting optimizer class')
        if self.config.use_8bit_adam:
            print('hit condition use 8bit adam')
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            self.optimizer_class = bnb.optim.AdamW8bit
        else:
            print('not using 8bit adam')
            self.optimizer_class = torch.optim.AdamW

        print('Getting optimizer')
        params_to_optimize = (
            itertools.chain(self.unet.parameters(),
            self.text_encoder.parameters()) if self.config.train_text_encoder else self.unet.parameters()
            )
        self.optimizer = self.optimizer_class(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        print('within __init__(), triggering self.generate_class_images() prior to making the dataloader')
        if self.config.do_prior_preservation:
            self.generate_class_images()

        print('Loading image dataset')
        self.train_dataset = DreamBoothDataset(
            instance_data_dir=self.config.instance_data_dir,
            instance_prompt=self.config.instance_prompt,
            class_data_dir=self.config.class_data_dir if self.config.do_prior_preservation else None,
            class_prompt=self.config.class_prompt,
            tokenizer=self.tokenizer,
            size=self.config.resolution,
            center_crop=self.config.center_crop,
            )



        collater = Collater(do_prior_preservation=self.config.do_prior_preservation, tokenizer=self.tokenizer)


        '''def collate_fn(examples):
            """
            Custom collate function to pass to the DataLoader. This concats class and instance examples
            for prior preservation to avoid doing two forward passes
            """
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if self.config.do_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch'''




        print('Setting up the DataLoader')
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True, collate_fn=collater, num_workers=1)

        print('Determining number of training steps')
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        print('len(self.train_dataloader):', len(self.train_dataloader))
        print('self.config.gradient_accumulation_steps:', self.config.gradient_accumulation_steps)
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        print('num_update_steps_per_epoch:', num_update_steps_per_epoch)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        print('max_train_steps:', self.config.max_train_steps)
        print('overrode_max_train_steps:', overrode_max_train_steps)

        print('Getting lr_scheduler')
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
            )

        print('Preparing unet, optimizer, train_dataloader, and lr_scheduler with accelerator')
        if self.config.train_text_encoder:
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        print('Setting up weight dtype')
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        print('Move text_encode and vae to correct accelerator device')
        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if not self.config.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        print('Recalculating total training steps and training epochs')
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.num_train_epochs = math.ceil(self.config.max_train_steps / num_update_steps_per_epoch)

        print('Initializing trackers and storing configuration')
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth", config=vars(self.config))

    def generate_class_images(self):
        print('='*50)
        print('Generating new class images for prior preservation')
        print('='*50)
        #if args.with_prior_preservation:
        class_images_dir = Path(self.config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        print(f'We want to create {self.config.num_class_images} total class images')
        print(f'There are currently {cur_class_images} in the class_images_dir {self.config.class_data_dir}')

        if cur_class_images < self.config.num_class_images:
            num_new_images = self.config.num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}")

            print(f'Generating {num_new_images} new images')
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(   # was previously DiffusionPipeline
                self.config.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=self.config.revision,
                use_auth_token=self.auth_token
            )
            pipeline.set_progress_bar_config(disable=True)
            print(f'Setting up prompt data set with class prompt: "{self.config.class_prompt}"')
            sample_dataset = PromptDataset(self.config.class_prompt, num_new_images)
            print('Setting up sample dataloader')
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.config.sample_batch_size)
            print('Preparing sample dataloader with accelerator')
            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)
            print(f'Accelerator device: {self.accelerator.device}')

            print('Now generating images')
            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
                    print(f'Generated image at {image_filename}')

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            print('Not generating images because there are already enough images in the class images directory')

            '''# Update these attributes now that we have class images in the directory
            print(f'Before, self.class_images_path: {self.class_images_path}')
            self.class_images_path = list(self.class_data_dir.iterdir())
            print(f'After, self.class_images_path: {self.class_images_path}')

            print(f'Before, self.num_class_images: {self.num_class_images}')
            self.num_class_images = len(self.class_images_path)
            print(f'After, self.num_class_images: {self.num_class_images}')

            print(f'Before, self._length: {self._length}')
            self._length = max(self.num_class_images, self.num_instance_images)
            print(f'After, self._length: {self._length}')'''

    # Train!
    def train(self):
        total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps

        print("***** Running training *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Num batches each epoch = {len(self.train_dataloader)}")
        print(f"  Num Epochs = {self.config.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.config.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(self.config.num_train_epochs):
            #print(f'Doing global step: {global_step} at time {datetime.datetime.now()}')
            self.unet.train()
            if self.config.train_text_encoder:
                #print(f'Training text_encoder for global step: {global_step}')
                self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                #print(f'Step: {step}')
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    #print('converting images to latent space')
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    #print('sampling noise to add to latents')
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    #print('Adding noise to the latents')
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    #print('Getting the text embedding for conditioning')
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    #print('predicting the noise residual')
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    #print('Getting the target for loss depending on the prediction type')
                    if self.noise_scheduler.predict_epsilon:
                    #if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    #elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    else:
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    #else:
                        #raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    if self.config.do_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.config.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    #print('Doing backward')
                    self.accelerator.backward(loss)
                    #print('Syncing gradients')
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if self.config.train_text_encoder
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
                    #print('Doing optimizer step')
                    self.optimizer.step()
                    #print('Doing lr_scheduler step')
                    self.lr_scheduler.step()
                    #print('Optimizer zero_grad')
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    print('Updating progress bar')
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.config.save_steps == 0:
                        if self.accelerator.is_main_process:
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                self.config.model_id,
                                unet=self.accelerator.unwrap_model(self.unet),
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                                revision=self.config.revision,
                                use_auth_token=self.auth_token
                            )
                            save_path = os.path.join(self.config.model_out_dir, f"checkpoint-{global_step}")
                            pipeline.save_pretrained(save_path)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.config.max_train_steps:
                    break

            self.accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_id,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.config.revision,
                use_auth_token=self.auth_token
            )
            print(f'Saving model to {self.config.model_out_dir}')
            pipeline.save_pretrained(self.config.model_out_dir)

            #if self.config.push_to_hub:
                #repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

        self.accelerator.end_training()
        print('Training done')

