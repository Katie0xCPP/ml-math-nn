# src/neural_net.py
import numpy as np

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, seed=0):
        rng = np.random.default_rng(seed)

        # Xavier/Glorot-style scaling for better initialization
        self.W1 = rng.normal(0, np.sqrt(2 / (input_dim + hidden_dim)), (hidden_dim, input_dim))
        self.b1 = np.zeros((hidden_dim, 1))

        self.W2 = rng.normal(0, np.sqrt(2 / (hidden_dim + output_dim)), (output_dim, hidden_dim))
        self.b2 = np.zeros((output_dim, 1))

        self.lr = learning_rate

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        # z: (output_dim, batch_size)
        z_shift = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        # X shape: (input_dim, batch_size)
        Z1 = self.W1 @ X + self.b1    # (hidden_dim, batch_size)
        A1 = self.relu(Z1)            # activation
        Z2 = self.W2 @ A1 + self.b2   # (output_dim, batch_size)
        A2 = self.softmax(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    @staticmethod
    def one_hot(y, num_classes):
        # y: (batch_size,)
        batch_size = y.shape[0]
        Y = np.zeros((num_classes, batch_size))
        Y[y, np.arange(batch_size)] = 1.0
        return Y

    def compute_loss(self, A2, Y):
        # Cross-entropy
        # A2, Y: (num_classes, batch_size)
        m = Y.shape[1]
        eps = 1e-12
        log_probs = np.log(A2 + eps)
        loss = -np.sum(Y * log_probs) / m
        return loss

    def backward(self, cache, Y):
        # Y: (num_classes, batch_size)
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        # dZ2 = A2 - Y  (gradient of loss wrt Z2 for softmax+cross-entropy)
        dZ2 = A2 - Y                               # (output_dim, batch_size)
        dW2 = (dZ2 @ A1.T) / m                     # (output_dim, hidden_dim)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Backprop through hidden layer
        dA1 = self.W2.T @ dZ2                      # (hidden_dim, batch_size)
        dZ1 = dA1 * self.relu_deriv(Z1)            # (hidden_dim, batch_size)
        dW1 = (dZ1 @ X.T) / m                      # (hidden_dim, input_dim)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        A2, _ = self.forward(X)
        return np.argmax(A2, axis=0)
