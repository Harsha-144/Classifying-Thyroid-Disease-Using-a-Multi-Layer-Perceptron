# Directory: model/utils.py
import numpy as np

def sigmoid(x, der=False):
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig) if der else sig

def load_weights(path="model/trained_weights.npz"):
    data = np.load(path)
    return data["w1"], data["w2"], data["b1"], data["b2"]

def predict(inputs, w1, w2, b1, b2):
    hidden = sigmoid(np.dot(w1, inputs) + b1)
    output = sigmoid(np.dot(w2, hidden) + b2)
    return output
