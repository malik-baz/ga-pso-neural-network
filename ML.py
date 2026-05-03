import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#  Data Loading and Preprocessing
X, y = load_digits(return_X_y=True)
y = y.reshape(-1, 1)

enc = OneHotEncoder(sparse_output=False)
y_onehot = enc.fit_transform(y)

X_train, X_test, y_train_raw, y_test_raw, y_train_onehot, y_test_onehot = train_test_split(
    X, y, y_onehot, test_size=0.2, random_state=42
)

# Data Normalization
X_train = X_train / 16.0
X_test = X_test / 16.0


#  Activation and Derivative Functions
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

#for back prob
def sigmoid_derivative(x):
    return x * (1 - x)



def relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x <= 0] = alpha
    return dx



def generate_wt(input_size, output_size, activation='relu'):
    """He initialization for ReLU/Leaky ReLU, Xavier for sigmoid"""
    if activation in ['relu', 'leaky_relu']:
        scale = np.sqrt(2.0 / input_size)
    else:
        scale = np.sqrt(1.0 / input_size)
    return np.random.randn(input_size, output_size) * scale


def f_forward(x, w1, w2):
    """Performs the feedforward pass using Leaky ReLU for hidden layer."""
    z1 = x.dot(w1)
    a1 = relu(z1)  # Hidden Layer (Leaky ReLU)

    z2 = a1.dot(w2)
    a2 = sigmoid(z2)  # Output Layer (Sigmoid)

    return a2, a1, z1


def loss(out, Y):
    """Cross-Entropy Loss """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-12
    out = np.clip(out, epsilon, 1 - epsilon)
    return -np.sum(Y * np.log(out)) / len(Y)


def predict(X, w1, w2):
    out, _, _ = f_forward(X, w1, w2)
    return np.argmax(out, axis=1)

#check if predict equal real then divide on the sum of true
def calculate_accuracy(X, y_true_raw, w1, w2):
    y_pred = predict(X, w1, w2)
    return np.sum(y_pred == y_true_raw.ravel()) / len(y_true_raw)


def back_prop(x, y_onehot, w1, w2, alpha):
    """Performs the backpropagation and updates weights."""
    a2, a1, z1 = f_forward(x, w1, w2)

    # FIX: Using cross-entropy derivative for output layer
    # For sigmoid + cross-entropy: d2 = a2 - y_onehot
    d2 = a2 - y_onehot

    # FIX: Proper gradient calculation for hidden layer
    # Using Leaky ReLU derivative
    d1 = np.multiply(d2.dot(w2.T), relu_derivative(z1))

    # FIX: Gradient calculation with momentum (optional but helpful)
    w1_adj = x.T.dot(d1)
    w2_adj = a1.T.dot(d2)

    # FIX: Weight update with gradient normalization
    w1_adj_norm = np.linalg.norm(w1_adj)
    w2_adj_norm = np.linalg.norm(w2_adj)

    if w1_adj_norm > 5:  # Gradient clipping
        w1_adj = w1_adj * (5 / w1_adj_norm)
    if w2_adj_norm > 5:
        w2_adj = w2_adj * (5 / w2_adj_norm)

    w1 -= alpha * w1_adj
    w2 -= alpha * w2_adj

    return w1, w2


#  Training Function
def train(X, y_onehot, y_true_raw, w1, w2, alpha, epochs):
    loss_list = []
    accuracy_list = []
    print_interval = 100

    print(f"\nTraining Architecture: Input({X.shape[1]}) -> Hidden({w1.shape[1]}) -> Output({y_onehot.shape[1]})")
    print(f"Activation: **Leaky ReLU** (Hidden) / Sigmoid (Output) | Loss: Cross-Entropy | Learning Rate: {alpha}")
    print("-" * 50)

    for i in range(epochs):
        out, _, _ = f_forward(X, w1, w2)

        loss_val = loss(out, y_onehot)
        loss_list.append(loss_val)

        acc_val = calculate_accuracy(X, y_true_raw, w1, w2)
        accuracy_list.append(acc_val)

        w1, w2 = back_prop(X, y_onehot, w1, w2, alpha)

        if i % print_interval == 0:
            print(f"Epoch {i:4d}, Loss: {loss_val:.4f}, Train Accuracy: {acc_val * 100:.2f}%")

    return w1, w2, loss_list, accuracy_list


#  Network Parameters and Execution
print("input the activation function,segmoid or relu")
a = input("")
print("input the number of layers,  16  or  32 or 64 or  128")
b =  int(input(""))
input_size = X_train.shape[1]
hidden_size = b
output_size = y_train_onehot.shape[1]
# Initialize weights with proper initialization
w1 = generate_wt(input_size, hidden_size, activation=a)
w2 = generate_wt(hidden_size, output_size, activation=a)

# learning rate
alpha = 0.1
epochs = 1000

print(f"--- Starting Training ({epochs} epochs) on Digits Dataset (Normalized) ---")
w1, w2, loss_list, accuracy_list = train(X_train, y_train_onehot, y_train_raw, w1, w2, alpha, epochs)
print("--- Training Complete ---")

#  Plot Loss and Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(loss_list)
ax1.set_title('Training Loss over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(accuracy_list)
ax2.set_title('Training Accuracy over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#  Evaluate on Test Set ---
accuracy = calculate_accuracy(X_test, y_test_raw, w1, w2)
#percantage of the correct
print(f"\n--- Final Evaluation ---")
print(f"Test set accuracy: {accuracy * 100:.2f}%")

# Check neuron activation statistics
_, a1, z1 = f_forward(X_train, w1, w2)
active_neurons = np.mean(a1 > 0)
print(f"Percentage of active  relu neurons: {active_neurons * 100:.1f}%")
