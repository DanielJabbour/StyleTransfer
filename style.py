from __future__ import print_function

import time

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from preprocess import *

def make_keras(content_path, style_path)
    #Defining variables in Keras backend
    content_arr, style_arr = process_images(content_path,style_path)
    content_image = backend.variable(content_arr)
    style_image = backend.variable(style_arr)

    #Temporary placeholder for combination of content and style image
    combined_image = backend.placeholder((1, 512, 512, 3))

    #Building single tensor containing style, content, and combination images suitable for Keras's VGG16
    input_tensor = backend.concatenate([content_image, style_image, combined_image], axis=0)

    #Accessing Keras's pretrained VGG16 model, set top to false as we are not concerned with classification
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    #Storing layer information in a dictionary
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    return layers
