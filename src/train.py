import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_digits_dataset
from neural_net import NeuralNet

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def main():
    # 1. Load data
    X_train, X_test, y_train, y_test, digits = load_digits_dataset()

    # Transpose to shape (features, samples) for our math
    X_train_T = X_train.T
    X_test_T = X_test.T

    input_dim = X_train_T.shape[0]   # 64
    hidden_dim = 32                  # you can tune this
    output_dim = 10                  # digits 0-9

    # 2. Create model
    model = NeuralNet(input_dim, hidden_dim, output_dim, learning_rate=0.5)

    # 3. Training hyperparameters
    num_epochs = 100
    batch_size = 64

    n_train = X_train.shape[0]
    num_classes = output_dim

    losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            if X_batch.shape[0] == 0:
                continue

            X_batch_T = X_batch.T                         # (input_dim, batch_size)
            Y_batch = model.one_hot(y_batch, num_classes) # (num_classes, batch_size)

            A2, cache = model.forward(X_batch_T)
            loss = model.compute_loss(A2, Y_batch)
            model.backward(cache, Y_batch)

        # Track performance per epoch
        y_train_pred = model.predict(X_train_T)
        y_test_pred = model.predict(X_test_T)

        train_acc = accuracy(y_train, y_train_pred)
        test_acc = accuracy(y_test, y_test_pred)

        # Compute loss on full train set (optional; approximate with last batch loss)
        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    # 4. Plot learning curves
    epochs = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()