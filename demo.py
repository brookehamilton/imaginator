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

def run_gradio(starting_image: PIL.Image.Image, prompt: str):
    images = run_pipeline(pipe,
                prompt=prompt,
                init_image=starting_image,
                strength=0.75,
                guidance_scale=7.5,
                negative_prompt=None,
                num_inference_steps=15,
                seed=102)

    return images[0]

demo = gr.Interface(
    fn=run_gradio,
    inputs=[gr.Image(type="pil"), "text"],
    #inputs=["text", "text", "text",  "text", "text", "number", "number",
            #gr.Slider(0.5, 1), gr.Slider(7, 8), "checkbox"],
    outputs=["image"],
    )

demo.launch(share=True)
