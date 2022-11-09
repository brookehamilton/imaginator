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
                strength: float,
                guidance_scale: float,
                num_inference_steps: int,
                seed: int,):

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

            gr.inputs.Slider(0.0, 1.0, step=0.05, default=0.75,             # strength
                label='Strength (how much to transform the starting image; higher values less faithful to starting image'),
            gr.inputs.Slider(6, 9, step=0.25, default=7.5,                  # guidance scale
                label='Guidance Scale (how much to match text prompt; higher values match text more)'),
            gr.Number(value=15, label='Num Inference Steps', precision=0),  # num_inference_steps
            gr.Number(value=102)                                              # seed
            ],
    outputs=["image"],
    examples=[['starting_images/sample_images/red_monster.png', 'photo of red hairy monster with three eyes, award winning photography, national geographic, nikon'],
    ['starting_images/sample_images/long_neck_monster.png', 'charcoal drawing of a monster with long neck and outstretched arms, trending on artstation']]
    )

demo.launch()
#demo.launch(share=True)
