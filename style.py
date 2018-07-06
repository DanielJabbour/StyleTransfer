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

layers = make_keras('./Images/content.jpg','./Images/style.jpg')

#We are minimizing a loss function consisting of content, style, and total variation hence we define 3 weights
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

#initialize total loss to 0
loss = backend.variable(0.)

#Use feature spaces provided by model layers to define loss functions

#We will obtain the content feature from layer block2_conv2 as follows from Johnson et al. (2016)

def content_loss():

    #Obtaining content and combined image features information from appropriate layer
    content_features = layers['block2_conv2'][0, :, :, :]
    combined_features = layers['block2_conv2'][2, :, :, :]

    #Computing scaled Euclidean distance between feature representations of the 2 images
    content_loss = backend.sum(backend.square(combined_features - content_features))
    loss += content_weight * content_loss)

    return loss

