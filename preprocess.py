from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def process_images(content_path, style_path, combined_image):
    #This function takes 2 image paths as an argument, one content and one style, and a white noise image
    #The images are formated into a numpy array to be used in a TensorFlow Graph
    #We then extract the appropriate features from a pretrained image classification model

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

    # Pass to tensorflow, use a white noise placeholder image for initial combination
    content_image = backend.variable(content_arr)
    style_image = backend.variable(style_arr)

    #Building single tensor containing style, content, and combination images suitable for Keras's VGG16
    input_tensor = backend.concatenate([content_image, style_image, combined_image], axis=0)

    #Accessing Keras's pretrained VGG16 model, set top to false as we are not concerned with classification
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    #Storing layer information in a dictionary
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    return layers