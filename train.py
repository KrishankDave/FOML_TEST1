import tensorflow as tf
from tensorflow import keras
import numpy as np

# Example: Simple data for binary classification
# Features: Shape (samples, features)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
# Labels: Shape (samples, 1)
y = np.array([0, 1, 1, 0])  # XOR problem

# Build model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, verbose=1)

# Save the model
model.save('my_model.h5')
print("Model saved as my_model.h5")
