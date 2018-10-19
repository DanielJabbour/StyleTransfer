import numpy as np

import tensorflow as tf
from keras import backend
from keras.models import Model

from preprocess import *

height = 512
width = 512
combined_image = backend.placeholder((1, height, width, 3))

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

layers = process_images('Images/content.jpg', 'Images/style.jpg', combined_image)

loss = tf.Variable(0.)

def content_loss(loss):

    #Obtaining content and combined image features information from appropriate layer
    content_features = layers['block2_conv2'][0, :, :, :]
    combined_features = layers['block2_conv2'][2, :, :, :]

    #Computing scaled Euclidean distance between feature representations of the 2 images
    content_loss = backend.sum(tf.square(combined_features - content_features))
    loss += content_weight * content_loss

    return loss

def style_loss(loss):

    #Style feature layers
    feature_layers = ['block1_conv2', 'block2_conv2',
                    'block3_conv3', 'block4_conv3',
                    'block5_conv3']

    channels = 3
    size = height * width

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

def total_variation_loss(loss):
    #Maybe compute this differently?

    a = backend.square(combined_image[:, :height-1, :width-1, :] - combined_image[:, 1:, :width-1, :])
    b = backend.square(combined_image[:, :height-1, :width-1, :] - combined_image[:, :height-1, 1:, :])
    total_variation_loss = backend.sum(backend.pow(a + b, 1.25))

    loss += total_variation_weight * total_variation_loss

    return loss

def compute_losses(loss):

    content = content_loss(loss)
    style = style_loss(content)
    total = total_variation_loss(style)

    return total

loss = compute_losses(loss)
gradients = backend.gradients(loss, combined_image)
loss_grad = [loss] + gradients
f_combined = backend.function([combined_image], loss_grad)

def losses(x):
    x = x.reshape((1, height, width, 3))
    outs = f_combined([x])
    loss_value = outs[0]

    return loss_value

def gradients(x):
    x = x.reshape((1, height, width, 3))
    outs = f_combined([x])
    grad_values = outs[1].flatten().astype('float64')

    return grad_values