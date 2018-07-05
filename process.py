from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

def process_images(content_path, style_path):

    #Loading and resizing images to width x height of 512 x 512
    content_image = Image.open(content_path)
    content_image = content_image.resize((512,512))

    style_image = Image.open(style_path)
    style_image = style_image.resize((512,512))

