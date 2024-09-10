# activations.py
import torch
import torch.nn as nn

def relu(x):
    return torch.relu(x)

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def softmax(x, dim=1):
    return torch.softmax(x, dim=dim)