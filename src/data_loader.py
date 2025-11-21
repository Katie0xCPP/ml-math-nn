import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_digits_dataset(test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize
    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, digits
