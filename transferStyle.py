import time
import numpy as np

from PIL import Image
from scipy.optimize import fmin_tnc

from preprocess import *
from loss import *

class Loss_function(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        loss_value = losses(x) 
        grad_values = gradients(x)

        self.loss_value = loss_value
        self.grad_values = grad_values

        return self.loss_value

    def grads(self, x):
        grad_values = np.copy(self.grad_values)

        self.loss_value = None
        self.grad_values = None

        return grad_values

def train():

    loss_function = Loss_function()
    initial_image = np.random.uniform(0, 255, (1, 512, 512, 3)) - 128.

    num_iterations = int(input("Enter the number of iterations to minimize loss function: "))
    print("Minimizing loss function")

    for i in range(num_iterations):
        print('Iteration', i, 'of', num_iterations)

        start_time = time.time()

        # Performing function minimization using scipy's built in fmn_tnc method
        result, current_min, data = fmin_tnc(loss_function.loss, initial_image.flatten(), fprime=loss_function.grads)

        print('Loss:', current_min, 'for iteration: ', i)
        end_time = time.time()

        print('Iteration %d finished at timestamp: %ds' % (i, end_time - start_time))

    #Postprocessing, configure dimensions and add the imagenet mean rgb values to return originally subtracted value
    result = result.reshape((512, 512, 3))
    result = result[:, :, ::-1]

    result[:, :, 0] += 103.939
    result[:, :, 1] += 116.779
    result[:, :, 2] += 123.68

    result = np.clip(x, 0, 255).astype('uint8')

    Image.fromarray(x)

train()