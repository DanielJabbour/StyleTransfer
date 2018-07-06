from PIL import Image
import numpy as np

def process_images(content_path, style_path):
    #This function takes 2 image paths as an argument, one content and one style
    #The images are formated into a numpy array to be used in a TensorFlow Graph

    #Loading and resizing images to width x height of 512 x 512
    content_image = Image.open(content_path)
    content_image = content_image.resize((512,512))

    style_image = Image.open(style_path)
    style_image = style_image.resize((512,512))

    #Convert image to numpy array and insert a new axis at position 0
    content_arr = np.asarray(content_image, dtype='float32')
    style_arr = np.asarray(style_image, dtype='float32')

    #Calculate mean RGB value
    content_rgb_avg = np.mean(content_arr, axis=(0,1))
    style_rgb_avg = np.mean(style_arr, axis=(0,1))

    #Numpy array shape is now 1 x 512 x 512 x 3
    content_arr = np.expand_dims(content_arr, axis=0)
    style_arr = np.expand_dims(style_arr, axis=0)

    #Subtract mean RGB value from content and style arrays
    content_arr[:, :, :, 0] -= content_rgb_avg[0]
    content_arr[:, :, :, 1] -= content_rgb_avg[1]
    content_arr[:, :, :, 2] -= content_rgb_avg[2]

    style_arr[:, :, :, 0] -= style_rgb_avg[0]
    style_arr[:, :, :, 1] -= style_rgb_avg[1]
    style_arr[:, :, :, 2] -= style_rgb_avg[2]

    #Invert last array dimension to convert from RGB to BGR
    content_arr = content_arr[:, :, :, ::-1]
    style_arr = style_arr[:, :, :, ::-1]

    return content_arr, style_arr

#Sample call
#content, style = process_images('./Images/content.jpg','./Images/style.jpg')