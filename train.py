import os
import cv2                      #process images
import numpy as np              #working with numpy arrays
import matplotlib.pyplot as plt #visualisation of digits
import tensorflow as tf


# load dataset
mnist = tf.keras.datasets.mnist
# load training and testing data
(x_train , y_train) , (x_test , y_test) = mnist.load_data()

# normalize the training data
x_train = tf.keras.utils.normalize(x_train, axis=1)
# normalize the testing data
x_test = tf.keras.utils.normalize(x_test, axis=1)

# define the neural network model
# create sequential model
model = tf.keras.models.Sequential()
# add a flatten layer to convert input images into 1D arrays
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# add a dense layer with ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# add another dense layer with ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# add the output layer with softmax activation for multiclass classification
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=3)

# save the trained model
model.save('handwritten.model')
