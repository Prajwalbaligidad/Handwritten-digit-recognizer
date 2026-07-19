import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Images Shape :", X_train.shape)
print("Training Labels Shape :", y_train.shape)

print("Testing Images Shape :", X_test.shape)
print("Testing Labels Shape :", y_test.shape)


# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Maximum Pixel Value :", X_train.max())
print("Minimum Pixel Value :", X_train.min())

# Flatten
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print("New Training Shape :", X_train.shape)
print("New Testing Shape :", X_test.shape)

# Create ANN model

model = Sequential()


# Input + First Hidden Layer

model.add(
    Dense(
        128,
        activation='relu',
        input_shape=(784,)
    )
)


# Second Hidden Layer

model.add(
    Dense(
        64,
        activation='relu'
    )
)


# Output Layer

model.add(
    Dense(
        10,
        activation='softmax'
    )
)


# Show model structure

model.summary()





model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


model.save("model/digit_model.h5")

print("Model trained and saved successfully!")