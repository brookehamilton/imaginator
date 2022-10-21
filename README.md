## Bringing Kids' Artwork to Life Using Stable Diffusion (img2img)

Brooke Hamilton  
brookehamilton@gmail.com 

This project uses [Stable Diffusion](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers:~:text=VRAM.%20See%20the-,model%20card,-for%20more%20information) via the [Huggingface diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers) to generate artwork based on kids' drawings.

Original:  
![original drawing](https://github.com/brookehamilton/stable-diffusion/blob/main/images/readme_images/original_monster.png?raw=true)

Output:  
![output drawing](https://github.com/brookehamilton/stable-diffusion/blob/main/images/readme_images/output_monster.png?raw=true)


## Getting Started

### HuggingFace setup
- If you don't have one yet, sign up for an account with Huggingface (including an [access token](https://huggingface.co/docs/hub/security-tokens))
- Go to the model card for Stable Diffusion and [accept the Stable Diffusion license](https://huggingface.co/CompVis/stable-diffusion-v1-4)

## Running on GPU vs. CPU
If you don't have access to a GPU, this can be run (slowly) on a CPU. On my Mac, it takes about 8 minutes to generate an image.  Another option is to use Google Colab (see below).

### Running in Google Colab
If you don't have access to a GPU environment, Google Colab is a great alternative. I have included a Google Colab notebook [here](https://colab.research.google.com/drive/1HWg40vl8Td4oNliS3XW6ouLeJMzUrjo0#scrollTo=MWFi2PNQkL6u).

Instructions for running in Google Colab:
- Click "Make a Copy" to save an editable copy of the notebook to your drive
- Connect to a GPU runtime (`Runtime -> Change runtime type`)
- Upload your starting images into the Colab filesystem using the GUI on the left side of the screen
- Make sure to run the notebook in order from the top cell downward so that all dependencies are installed correctly

### Virtual Environment
I recommend using the virtual environment of your choice, such as virtualenv.

Example:
```
cd <this project's directory>
virtualenv sd_env
source sd_env/bin/activate
pip install -r requirements.txt
```

## Creating images
To run at the command line:
```
python3 run.py --init_image_path='images/sample_images/red_monster.png' --prompt='Red monster, trending on artstation'
```

