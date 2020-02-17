import os

import numpy as np
import h5py
import scipy
import random
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from scipy import ndimage

from src.little_networks.network_build_utils import *

def load_data():
    train_dataset = h5py.File('train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_dataset = h5py.File('test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def build():

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    return train_x,train_y,test_x,test_y,classes

def models(train_x,train_y):
    layers_dims = [12288, 20, 7, 5, 1]
    return model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)

def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 10000, print_cost=False):
    parameters = initialize_parameters(layers_dims)
    costs = []
    for i in range(0,num_iterations):
        AL, caches = model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = model_backward(AL, Y, caches)
        parameters = update_parameters(grads, parameters, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters



def judge(parameters,classes):

    fname = "images/"
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((64*64*3,1))
    my_image = my_image/255.
    my_predicted_image = predict(my_image, 1, parameters)

    plt.imshow(image)
    plt.show()
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
