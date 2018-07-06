from __future__ import print_function

import time

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from preprocess import *

content_arr, style_arr = process_images('./Images/content.jpg','./Images/style.jpg')