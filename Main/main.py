# import cv2

# image_file = '..data/because.jpeg'
# img = cv2.imread(image_file)


# import cv2
# from tensorflow.keras.models import load_model

# model = load_model('trained_model.h5')

# # Load the preprocessed image and the trained model
# img = cv2.imread('../data/because.jpeg', cv2.IMREAD_GRAYSCALE)
# model = tf.keras.models.load_model('trained_model.h5')

# # Reshape the image to the expected input shape of the model
# img = cv2.resize(img, (28, 28))
# img = img.reshape(1, 28, 28, 1)

# # Normalize the image
# img = img / 255.0

# # Predict the output
# predicted_output = model.predict(img)
# predicted_output = np.argmax(predicted_output)
# predicted_output = chr(predicted_output + 65)

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model
# from imageManipulation.ipynb import imageManipulation

# Load pre-trained Keras model
model = load_model('trained_model.h5')

# Load image using OpenCV
img = cv2.imread('../data/complex1.jpg')

image = imageManipulation(img)

prediction = model.predict(np.array([image]))

class_label = np.argmax(prediction)

print("Prediction Class Label: ", class_label)