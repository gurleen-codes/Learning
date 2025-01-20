# Micrograd Implementation

This is my personal re-implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), a lightweight framework for autograd and simple neural networks.

## Features
- Implements a basic `Value` class for automatic differentiation.
- Contains `Neuron`, `Layer`, and `MLP` classes for building simple neural networks.
- Allows forward and backward passes for gradients computation.

## Usage
Here's an example of how to use this implementation:

```python
from micrograd import MLP

# Create a simple MLP with 2 inputs, a hidden layer of 3 neurons, and 1 output
model = MLP(2, [3, 1])

# Input data
x = [1.0, 2.0]

# Get the output
output = model(x)
print(output)

