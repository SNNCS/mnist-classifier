import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the MNIST dataset
(x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()

# Data preprocessing
x_train = x_train / 255.0  # Normalize pixel values to the range [0, 1]
x_valid = x_valid / 255.0  # Apply the same normalization to the validation set

# Reshape data (batch size, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)  # MNIST images are 28x28 grayscale (single channel)
x_valid = x_valid.reshape(-1, 28, 28, 1)

# Build the model
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),  # Flatten each image into a 1D array
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  # Softmax activation for multi-class classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_valid, y_valid))

plt.figure(figsize=(12, 5))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')

plt.show()
