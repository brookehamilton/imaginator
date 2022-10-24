## Bringing Kids' Artwork to Life Using Stable Diffusion (img2img)

Brooke Hamilton  
brookehamilton@gmail.com 

This project uses [Stable Diffusion](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers:~:text=VRAM.%20See%20the-,model%20card,-for%20more%20information) via the [Huggingface diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers) to generate artwork based on kids' drawings.

Original:  
![original drawing](https://github.com/brookehamilton/stable-diffusion/blob/main/starting_images/readme_images/red_monster.png?raw=true)

Output:  
![output drawing](https://github.com/brookehamilton/stable-diffusion/blob/main/starting_images/readme_images/red_monster_output.png?raw=true)


## Getting Started

### HuggingFace setup
- If you don't have one yet, sign up for an account with Huggingface (including an [access token](https://huggingface.co/docs/hub/security-tokens))
- Go to the model card for Stable Diffusion and [accept the Stable Diffusion license](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Access token
If desired, you can define your Huggingface access token as the environment variable `HUGGINGFACE_ACCESS_TOKEN`, such as in your `.zshrc` as:
`export HUGGINGFACE_ACCESS_TOKEN={your token here}`
If this variable is not found, the code will prompt the user for the token each time.

### Running on GPU vs. CPU
If you don't have access to a GPU, this can be run (slowly) on a CPU. On my Mac, it takes about 4 minutes to generate an image around 700x500.  Another option is to use Google Colab (see below).

### Running in Google Colab
If you don't have access to a GPU environment, Google Colab is a great alternative. I have included a Google Colab notebook [here](https://colab.research.google.com/drive/18Iza0DAxRYWmZbQJ51TJx69HSLS6Jf9j?usp=sharing).

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
python3 run.py --init_image_path='starting_images/sample_images/red_monster.png' --prompt='photo of red hairy monster with three eyes, award winning photography, national geographic, nikon' --seed=838120
```

### Image size
If the input image is too large, you may not have enough memory to run the script.  On my machine, I get this error:
`UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`
In this case, resize the initial image and try again.

### Prompt engineering
The best results come from adding additional, description terms and phrases to the prompt.
Suggestions:
- unreal engine
- national geographic
- trending on artstation
- nikon
- hyperrealistic
- award winning photo
- ((name of artist, e.g. Van Gogh))

