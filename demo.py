"""
Imaginator - Demo

Author: Brooke Hamilton
brookehamilton@gmail.com

This script generates a user-friendly demo using Gradio
"""
import gradio as gr
from imaginator.pipeline_utils import create_pipeline, run_pipeline
import PIL

pipe = create_pipeline()

def run_gradio(starting_image: PIL.Image.Image,
                prompt: str,
                seed: int,
                strength: float,
                guidance_scale: float,
                num_inference_steps: int):

    images = run_pipeline(pipe,
                prompt=prompt,
                init_image=starting_image,
                strength=strength,
                guidance_scale=guidance_scale,
                negative_prompt=None,
                num_inference_steps=num_inference_steps,
                seed=seed)

    return images[0]

demo = gr.Interface(
    fn=run_gradio,
    inputs=[gr.Image(type="pil"),                                           # starting_image
            "text",                                                         # prompt
            gr.Number(value=455),                                           # seed
            gr.inputs.Slider(0.0, 1.0, step=0.05, default=0.75,             # strength
                label='Strength (how much to transform the starting image; higher values less faithful to starting image'),
            gr.inputs.Slider(6, 9, step=0.25, default=7.5,                  # guidance scale
                label='Guidance Scale (how much to match text prompt; higher values match text more)'),
            gr.Number(value=15, label='Num Inference Steps', precision=0)   # num_inference_steps
            ],
    outputs=["image"],
    )

demo.launch()
#demo.launch(share=True)
