# Neural Network From Scratch in NumPy

This project implements a fully connected neural network entirely from first principles using NumPy. 
All forward propagation, backward propagation, gradients, and parameter updates are manually coded without using deep learning frameworks such as TensorFlow or PyTorch.

The model is trained on the scikit-learn digits dataset and learns to classify handwritten digits (0–9).  
Training accuracy, test accuracy, and loss curves are generated using matplotlib.

---

## Repository Structure

ml-math-nn/
│
├── src/
│ ├── data_loader.py # Loads and normalizes digits dataset
│ ├── neural_net.py # Core neural network (forward + backward)
│ └── train.py # Training loop and metrics
│
├── data/ # Optional dataset storage
├── .venv/ # Virtual environment
└── README.md


---

## Mathematical Formulation

The neural network architecture is: Input (64) → Hidden Layer (ReLU) → Output Layer (Softmax, 10 classes)

### Forward Pass

$$
Z_1 = W_1 X + b_1
$$

$$
A_1 = \max(0, Z_1)
$$

$$
Z_2 = W_2 A_1 + b_2
$$

$$
A_2 = \text{softmax}(Z_2)
$$

### Cross-Entropy Loss

$$
\mathcal{L} = -\frac{1}{m}
\sum_{i=1}^{m} \sum_{k=1}^{10}
y_k^{(i)} \log A_{2,k}^{(i)}
$$

### Backward Pass

$$
dZ_2 = A_2 - Y
$$

$$
dW_2 = \frac{1}{m} dZ_2 A_1^T,
\qquad
db_2 = \frac{1}{m} \sum_i dZ_2
$$

$$
dA_1 = W_2^T dZ_2,
\qquad
dZ_1 = dA_1 \odot \mathbf{1}(Z_1 > 0)
$$

$$
dW_1 = \frac{1}{m} dZ_1 X^T,
\qquad
db_1 = \frac{1}{m} \sum_i dZ_1
$$

Parameters are updated with gradient descent:

$$
W := W - \eta dW,
\qquad
b := b - \eta db
$$

---

## Features

- Full neural network implemented manually in NumPy
- ReLU activation, softmax output, cross-entropy loss
- Backpropagation implemented from scratch
- Mini-batch stochastic gradient descent
- Tracks training and test accuracy
- Plots loss and accuracy curves using matplotlib

---

## How to Run

```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install numpy matplotlib scikit-learn

# Run the training script
python -m src.train

Requirements

Python 3.10+

NumPy

Matplotlib

scikit-learn
