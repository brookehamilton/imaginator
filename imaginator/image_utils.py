"""
Stable Diffusion - Pipeline Utility Functions

Brooke Hamilton
brookehamilton@gmail.com

Helper scripts for images
"""

from PIL import Image

def resize_image(init_image, desired_max_dimension=750):
    """
    Proportionally resize an image based on the pixels for the desired max dimension (length or width).
    This will preserve the aspect ratio.

    desired_max_dimension: size in pixels of longest desired dimension. For example, if set to 500, this will resize the image
    such that whichever side is longer is resized to 500, and the other side is scaled accordingly to maintain
    aspect ratio

    N.B. Stable diffusion does not perform well on images scaled much smaller than 500px
    """
    longest_dim = 'width' if init_image.width >= init_image.height  else 'height'

    if longest_dim == 'width':
        ratio_of_resize = desired_max_dimension/init_image.width
        desired_height = int(ratio_of_resize * init_image.height)
        resized_image = init_image.resize((desired_max_dimension, desired_height))
    else:
        ratio_of_resize = desired_max_dimension/init_image.height
        desired_width = int(ratio_of_resize * init_image.width)
        resized_image = init_image.resize((desired_width, desired_max_dimension))

    #(f'Inside resize_image(). Resized init image to {desired_max_dimension} pixels on longest side')
    #(f'Inside resize_image(). Actual resized image final dimensions are w: {resized_image.width}, h: {resized_image.height}')
    return resized_image

def image_grid(imgs, rows, cols):
    """
    Display a group of images as a grid, e.g. 3x3
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
