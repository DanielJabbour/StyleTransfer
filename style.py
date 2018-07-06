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

#Use feature spaces provided by model layers to define loss functions

#We will obtain the content feature from layer block2_conv2 as follows from Johnson et al. (2016)

def content_loss(initial_loss):

    #Obtaining content and combined image features information from appropriate layer
    content_features = layers['block2_conv2'][0, :, :, :]
    combined_features = layers['block2_conv2'][2, :, :, :]

    #Computing scaled Euclidean distance between feature representations of the 2 images
    content_loss = backend.sum(backend.square(combined_features - content_features))
    loss += content_weight * content_loss)

    return loss

def style_loss(initial_loss):

    #Constants for style loss (temp)
    feature_layers = ['block1_conv2', 'block2_conv2',
                    'block3_conv3', 'block4_conv3',
                    'block5_conv3']

    channels = 3
    size = 512 * 512
    
    #Computing gram matrix of style image
    style_features = backend.batch_flatten(backend.permute_dimensions(style, (2, 0, 1)))
    style_gram = backend.dot(style_features, backend.transpose(style_features))

    #Computing gram matrix of combined image
    combined_features = backend.batch_flatten(backend.permute_dimensions(combined, (2, 0, 1)))
    combined_gram = backend.dot(combined_features, backend.transpose(combined_features))
    
    back_sum = backend.sum(backend.square(style_gram - combined_gram)) / (4. * (channels ** 2) * (size ** 2))

    for layer_name in feature_layers:

        #Obtaining appropriate feature layers
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combined_features = layer_features[2, :, :, :]

        #Compute gram matricies for style and combined images
        style_flattened = backend.batch_flatten(backend.permute_dimensions(style_features, (2, 0, 1)))
        style_gram = backend.dot(style_flattened, backend.transpose(style_flattened))

        combined_flattened = backend.batch_flatten(backend.permute_dimensions(combined_features, (2, 0, 1)))
        combined_gram = combined_gram = backend.dot(combined_flattened, backend.transpose(combined_flattened))

        #Compute current style loss
        style_loss = backend.sum(backend.square(style_gram - combined_gram)) / (4. * (channels ** 2) * (size ** 2))

        loss += style_loss * (style_weight / len(feature_layers))

    return loss

def total_variation_loss(initial_loss):
    combined_image = backend.placeholder((1, 512, 512, 3))

    a = backend.square(combined_image[:, :height-1, :width-1, :] - combined_image[:, 1:, :width-1, :])
    b = backend.square(combined_image[:, :height-1, :width-1, :] - combined_image[:, :height-1, 1:, :])
    total_variation_loss = backend.sum(backend.pow(a + b, 1.25))

    loss += total_variation_weight * total_variation_loss

    return loss

def compute_losses():
    loss = backend.variable(0.)

    content_loss = content_loss(loss)
    style_loss = style_loss(content_loss)
    total_variation_loss = total_variation_loss(style_loss)

    return total_variation_loss