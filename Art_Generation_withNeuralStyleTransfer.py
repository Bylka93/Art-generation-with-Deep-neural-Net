import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
from neural_style import *

# Reset the graph
tf.reset_default_graph()
# Start interactive session
sess = tf.InteractiveSession()
# load the content image, the one we want to style
content_image = plt.imread("louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)
# load the style image
style_image = plt.imread("monet.jpg")
style_image = reshape_and_normalize_image(style_image)
# initial initialization of the generated image, the one we will
# optimze
generated_image = generate_noise_image(content_image)
# load the vgg19 model
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
# and the total cost
J = total_cost(J_content, J_style, alpha=10, beta=40)
# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

# define the general model


def model_nn(sess, input_image, num_iterations=200):

    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        sess.run(train_step)

        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('generated_image.jpg', generated_image)

    return generated_image


# and now the thing is here
model_nn(sess, generated_image)
