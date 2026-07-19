import numpy as np
import matplotlib.pyplot as plt
import random


from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model("model/digit_model.h5")

# Load the test dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess
X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], 784)

# Select an image

image_index = random.randint(0, 9999)

# Predict
prediction = model.predict(X_test[image_index].reshape(1, 784))

# Get predicted digit
predicted_digit = np.argmax(prediction)

# Print results
print("Image Index     :", image_index)
print("Predicted Digit :", predicted_digit)
print("Actual Digit    :", y_test[image_index])

# Display image
image = X_test[image_index].reshape(28, 28)

plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {predicted_digit} | Actual: {y_test[image_index]}")
plt.axis('off')
plt.show()