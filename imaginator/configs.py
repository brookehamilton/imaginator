"""
Stable Diffusion - Configuration classes

Brooke Hamilton
brookehamilton@gmail.com

This script has configuration classes used to specify parameters for image creation
"""
from dataclasses import dataclass

@dataclass
class InferenceConfig():
    prompt: str
    strength: float
    guidance_scale: float
    negative_prompt: str
    num_inference_steps: int
    seed: int

