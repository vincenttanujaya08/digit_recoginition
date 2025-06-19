"""
Simple Neural Network for MNIST Digit Recognition (from scratch with NumPy)

This script implements a fully connected neural network with:
- Input layer: 784 neurons (28×28 flattened pixels)
- 1 hidden layer: 128 neurons (ReLU activation)
- Output layer: 10 neurons (softmax for 10 digit classes)

Requirements:
- numpy
- tensorflow (only for loading MNIST data via keras.datasets)

You can install dependencies via:
    pip install numpy tensorflow

Usage:
    python mnist_nn_numpy.py
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -------------------------------
# 1. Load and preprocess MNIST
# -------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten 28×28 images → vectors of length 784, and normalize to [0,1]
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0  # shape (60000, 784)
x_test  = x_test.reshape(-1, 28*28).astype(np.float32)  / 255.0  # shape (10000, 784)

# Convert labels to one-hot vectors (shape: (n_samples, 10))
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# -------------------------------
# 2. Model Hyperparameters
# -------------------------------
input_size   = 784     # 28×28 pixels
hidden_size  = 128
output_size  = 10      # digits 0–9

learning_rate = 0.1
epochs        = 10
batch_size    = 64

# -------------------------------
# 3. Weight & Bias Initialization
# -------------------------------
# He initialization for hidden layer (ReLU), Xavier for output (softmax)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
b2 = np.zeros((1, output_size))

# -------------------------------
# 4. Activation Functions
# -------------------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    """ Numerically stable softmax """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# -------------------------------
# 5. Loss Function: Categorical Cross-Entropy
# -------------------------------
def compute_loss(y_true, y_pred):
    """
    y_true: one-hot shape (n_samples, 10)
    y_pred: probabilities shape (n_samples, 10)
    Returns average cross-entropy loss over the batch.
    """
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# -------------------------------
# 6. Training Loop
# -------------------------------
num_train = x_train.shape[0]
num_batches = num_train // batch_size

for epoch in range(1, epochs + 1):
    # Shuffle training data at the beginning of each epoch
    indices = np.random.permutation(num_train)
    x_train = x_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0.0

    for i in range(num_batches):
        start = i * batch_size
        end   = start + batch_size

        # ----- Forward Pass -----
        X_batch = x_train[start:end]          # shape: (batch_size, 784)
        Y_batch = y_train[start:end]          # shape: (batch_size, 10)

        # 1. Hidden layer
        Z1 = np.dot(X_batch, W1) + b1         # shape: (batch_size, 128)
        A1 = relu(Z1)                         # shape: (batch_size, 128)

        # 2. Output layer
        Z2 = np.dot(A1, W2) + b2              # shape: (batch_size, 10)
        A2 = softmax(Z2)                      # shape: (batch_size, 10)

        # Compute loss (for monitoring)
        batch_loss = compute_loss(Y_batch, A2)
        epoch_loss += batch_loss

        # ----- Backward Pass -----
        # Gradient of loss w.r.t. Z2 (softmax + cross-entropy)
        dZ2 = (A2 - Y_batch) / batch_size     # shape: (batch_size, 10)

        # Gradients for W2 and b2
        dW2 = np.dot(A1.T, dZ2)               # shape: (128, 10)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # shape: (1, 10)

        # Backprop into hidden layer
        dA1 = np.dot(dZ2, W2.T)               # shape: (batch_size, 128)
        dZ1 = dA1 * relu_derivative(Z1)       # shape: (batch_size, 128)

        # Gradients for W1 and b1
        dW1 = np.dot(X_batch.T, dZ1)          # shape: (784, 128)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # shape: (1, 128)

        # ----- Update Weights & Biases -----
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Average loss for the epoch
    epoch_loss /= num_batches
    print(f"Epoch {epoch}/{epochs} — Loss: {epoch_loss:.4f}")

# -------------------------------
# 7. Evaluate on Test Set
# -------------------------------
Z1_test = np.dot(x_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

y_pred_labels = np.argmax(A2_test, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_labels == y_true_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 8. Simpan Bobot ke File
# -------------------------------
np.savez('mnist_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Bobot model tersimpan ke 'mnist_weights.npz'")