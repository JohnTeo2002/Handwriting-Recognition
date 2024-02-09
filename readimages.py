import os
import cv2                      #process images
import numpy as np              #working with numpy arrays
import matplotlib.pyplot as plt #visualisation of digits
import tensorflow as tf

import Images

# Load pre-trained model
model = tf.keras.models.load_model('handwritten.model')

# Initialise image number counter
image_number = 1

# Loop through images until no more files are found
while os.path.isfile(f"Images/digit{image_number}.png"):
    try:
        # Read image and convert it to grayscale
        img = cv2.imread(f"Images/digit{image_number}.png")[:,:,0]
        # Invert the image and convert it to a numpy array
        img = np.invert(np.array([img]))
        # Predict using model
        prediction = model.predict(img)
        # Print predicted digit
        print(f"This digit is probably {np.argmax(prediction)}")
        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1

