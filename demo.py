"""
Imaginator - Demo

Author: Brooke Hamilton
brookehamilton@gmail.com

This script generates a user-friendly demo using Gradio
"""
import gradio as gr
from imaginator.pipeline_utils import create_pipeline, run_pipeline, resize_image
import PIL
from imaginator.configs import InferenceConfig
from imaginator.image_run import ImageRun

# Create the pipeline
pipe = create_pipeline()
#pipe.safety_checker = lambda images, clip_input: (images, False)

preset_dict = {
    'None': '',
   'Stuffed Animal': ', photo of cute hairy stuffed animal, award winning photography, national geographic, nikon',
   'Plastic Toy': ', photo of cute plastic toy, product image, high resolution',
   'Photorealistic Human': ''', hyperrealistic photo,
        realistic proportions, highly detailed, smooth, sharp focus, 8k, ray tracing,
        digital painting, concept art illustration, by artgerm, trending on artstation, nikon d850''',
   'Cute Japanese Style': ', kawaii Japenese cartoon style, cute, anime, sanrio, pokemon, studio ghibli, character art'
}

def run_gradio(starting_image: PIL.Image.Image,
                prompt: str,
                preset: str,
                strength: float,
                guidance_scale: float,
                num_inference_steps: int,
                seed: int,
                safety_checker: bool):
    # Set up inference config
    config_dict = {'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'negative_prompt': None,
            'num_inference_steps': num_inference_steps,
            'seed': seed
            }
    inference_config = InferenceConfig(**config_dict)

    # Run image creation
    Run = ImageRun(run_config = inference_config, pipeline=pipe, resize_pixels=500, safety_checker=safety_checker)
    Run.load_init_image_from_PIL(starting_image)
    Run.create_image()

    return Run.image


demo = gr.Interface(
    fn=run_gradio,
    inputs=[gr.Image(type="pil"),                                           # starting_image
            "text",                                                         # prompt
            gr.inputs.Dropdown(choices=list(preset_dict.keys()), default='None', label='Preset Style (Optional)'),   # preset
            gr.inputs.Slider(0.0, 1.0, step=0.05, default=0.75,             # strength
                label='Strength (how much to transform the starting image; higher values less faithful to starting image'),
            gr.inputs.Slider(6, 20, step=0.5, default=7.5,                  # guidance scale
                label='Guidance Scale (how much to match text prompt; higher values match text more)'),
            gr.Number(value=15, label='Num Inference Steps', precision=0),  # num_inference_steps
            gr.Number(value=102),                                              # seed
            gr.Checkbox(value=True, label='NSFW Safety Checker')],
    outputs=["image"],
    examples=[['starting_images/sample_images/red_monster.png', 'photo of red hairy monster with three eyes, award winning photography, national geographic, nikon'],
    ['starting_images/sample_images/long_neck_monster.png', 'charcoal drawing of a monster with long neck and outstretched arms, trending on artstation']]
    )

demo.launch()
#demo.launch(share=True)
