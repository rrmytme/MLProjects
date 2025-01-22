import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, utils
import os

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize the images to a range of 0 to 1
train_images = utils.normalize(train_images, axis=1)
test_images = utils.normalize(test_images, axis=1)

# Create the model
model = models.Sequential()

# Add the input layer and flatten the images to a 1D array of 784 elements (28x28)
model.add(layers.Flatten(input_shape=(28, 28)))

# Add the hidden layers with 128 neurons and ReLU activation function  
model.add(layers.Dense(units = 128, activation='relu'))
model.add(layers.Dense(units = 128, activation='relu'))

# Add the output layer with 10 neurons (0-9) and softmax activation function
model.add(layers.Dense(units = 10, activation='softmax'))

# Compile the model with the Adam optimizer and sparse categorical crossentropy loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with the training images and labels
model.fit(train_images, train_labels, epochs=3)

# Evaluate the model with the test images and labels
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')
print(f'Test loss: {test_loss}')

# Save the model
model.save('digits-recognition.h5')

# Load the model
#model = models.load_model('digits-recognition.h5')

os.chdir('neural-networks/')

for x in range(1,4):
    # Load an image
    image = cv.imread(f'{x}.png', cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28, 28))
    image = np.invert(np.array([image]))

    # Predict the image
    prediction = model.predict(image)
    print(f'Prediction: {np.argmax(prediction)}')

    # Display the image
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()
