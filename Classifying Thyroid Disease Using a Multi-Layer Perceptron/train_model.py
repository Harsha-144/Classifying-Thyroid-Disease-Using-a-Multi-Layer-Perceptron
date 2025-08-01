# Directory: train_model.py
import numpy as np
from model.utils import sigmoid

# Read training data
with open("data/ann-train.txt") as f:
    inputs_array = []
    outputs_array = []
    for line in f:
        i = line[:-5]

        if int(line[-4]) == 3:
            outputs_array.append([0, 0, 1])
        elif int(line[-4]) == 2:
            outputs_array.append([0, 1, 0])
        else:
            outputs_array.append([1, 0, 0])

        inputs_array.append([float(x) for x in i.split()])

inputs = np.array(inputs_array)
outputs = np.array(outputs_array)

# Parameters
p = 10  # Hidden layer neurons
q = 3   # Output layer classes
learning_rate = 1 / 5

# Initialize weights
w1 = 2 * np.random.rand(p, 21) - 1
b1 = 2 * np.random.rand(p) - 1
w2 = 2 * np.random.rand(q, p) - 1
b2 = 2 * np.random.rand(q) - 1

# Training
epochs = 150
for epoch in range(epochs):
    sum_error = 0
    for I in range(len(inputs)):
        x = inputs[I]
        z1 = sigmoid(np.dot(w1, x) + b1)
        y = sigmoid(np.dot(w2, z1) + b2)

        delta_output = (outputs[I] - y) * sigmoid(y, der=True)
        delta_hidden = np.dot(delta_output, w2) * sigmoid(z1, der=True)

        w2 += np.array([learning_rate * delta_output]).T * z1
        b2 += learning_rate * delta_output

        w1 += np.array([learning_rate * delta_hidden]).T * x
        b1 += learning_rate * delta_hidden

        sum_error += np.mean(abs(outputs[I] - y))

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Error: {sum_error / len(inputs):.4f}")

# Save weights
np.savez("model/trained_weights.npz", w1=w1, w2=w2, b1=b1, b2=b2)