import os
import cv2                      #process images
import numpy as np              #working with numpy arrays
import matplotlib.pyplot as plt #visualisation of digits
import tensorflow as tf


# load dataset
mnist = tf.keras.datasets.mnist
# load training and testing data
(x_train , y_train) , (x_test , y_test) = mnist.load_data()


model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
