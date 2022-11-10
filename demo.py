"""
Imaginator - Demo

Author: Brooke Hamilton
brookehamilton@gmail.com

This script generates a user-friendly demo using Gradio
"""
import gradio as gr
from imaginator.pipeline_utils import create_pipeline, run_pipeline, resize_image
import PIL

pipe = create_pipeline()
pipe.safety_checker = lambda images, clip_input: (images, False)

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
                seed: int,):

    starting_image = resize_image(starting_image)
    preset_text = preset_dict[preset]
    print(f'Using preset_text: {preset_text}')
    prompt += preset_text
    print(f'Using prompt: {prompt}')
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
            gr.inputs.Dropdown(choices=list(preset_dict.keys()), default='None', label='Preset Style (Optional)'),   # preset
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
