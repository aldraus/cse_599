from numba import njit, prange
import numpy as np
from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.activation = []
    def forward(self, data):

        forwardvals = np.copy(data)
        self.activation = forwardvals
        forwardvals[forwardvals <= 0] *= self.slope
        return forwardvals

    def backward(self, previous_partial_gradient):
        # TODO

        grads = previous_partial_gradient.copy()
        grads[self.activation <= 0] *= self.slope
        return grads

        # return None
