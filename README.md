# StyleTransfer

Artistic style transfer is the process of combining the content of one image into the style of another through the use of convolutional neural networks. The following image summarizes the concept perfectly.

<p align="center">
  <img src="Images/stanford.png?raw=true" title="Starry Night combined with Stanford's Hoover Tower, image taken from Stanford's CS231n course" alt="Style Transfer Image"/>
</p>

This project was built using the Numpy, Tensorflow and Keras libraries, and trained on an AWS EC2 GPU instance. You can clone this repository and give it a try with your own images! Depending on your computational power though, you might be waiting a long time before you get your new art.

## Data Preprocessing
Before performing any training, the content and style images have to be formatted into appropriate Numpy arrays to be used in a TensorFlow Graph, as well as slightly modified to achieve better results. This is accomplished through the "preprocess" script where images are converted into a 1 x 512 x 512 x 3 array with the mean RGB values subtracted, then inverted to follow a BGR colour scheme.

## Content and Style Representations

In order to accomplish this, we require a representation of the difference in the content of the 2 images, and the style of the 2 images. The representations of the content and style can be obtained by examining the intermediate layers of an image classification network. This is because an image classifying network needs to develop an understanding of an image in order to be able to classify it. Convolutional neural networks are excellent at achieving this as they are good at creating complex feature representations of each image. For this project, the VGG19 network architecture was used for image classification. Note that the final layers of the network were removed in our case as we are not concerned with the classification of the images, but rather the internal features defined by the network, which are then extracted as feature maps.

## Defining and Computing Content and Style Losses

Content loss was simply defined as the euclidean distance between our content image, and the base input image. Style loss was defined similarly, however rather than taking the direct euclidean distance, we define the style representation by taking the difference between the Gram matrix of the input and style image. The losses are then minimized using gradient descent, where the mean squared distance between the feature map of the input and that of the style image is minimized. It is through this procedure that we transform our base input image (in this case, initially a white noise image) into the combination of the given content and style.

## Acknowledgements

I would like to thank Stanford University's excellent academic documentation. A majority of the basis knowledge for this project was established thanks to their CS231n course on Convolutional Neural Networks.

A great resource for learning to use TensorFlow, a machine learning library, can be found on their official site here: https://www.tensorflow.org/tutorials/

You can learn more about this project by reading the original publication from Leon A. Gatys on neural style transfer here: https://arxiv.org/pdf/1508.06576.pdf

In the future, I would like to implement the feed-forward method to accomplish faster and more artistic results, introduced by Justin Johnson as published here: https://arxiv.org/pdf/1603.08155.pdf

