import numpy as np
from numba import njit, prange, jit
from .layer import Layer


class SigmoidLayer(Layer):
    def __init__(self, parent=None):
        super(SigmoidLayer, self).__init__(parent)
        self.activation = []
    def forward(self, data):
        activation = data.copy()
        self.activation = activation
        activation = 1/(1 + np.exp(-activation))
        return activation


    def backward(self, previous_partial_gradient):
        grads = previous_partial_gradient.copy()
        s = 1 / (1 + np.exp(-self.activation))
        grads = previous_partial_gradient * s * (1 - s)
        return grads





