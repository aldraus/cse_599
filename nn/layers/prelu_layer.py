import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.activation = []
        self.size = size
    def forward(self, data):
        activation = np.copy(data)
        self.activation = activation
        if self.size == 1:
            activation[activation <= 0] *= self.slope.data
        else:
            reshapedactivation = activation.reshape(-1, activation.shape[1], order ='C')
            abovezero = reshapedactivation.copy()
            abovezero[abovezero<0] = 0
            underzero = reshapedactivation.copy()
            underzero[underzero>0] = 0
            underzero =  underzero * self.slope.data
            activation_sum = underzero + abovezero
            activation = np.reshape(activation_sum,(np.shape(activation)))
        return activation

    def backward(self, previous_partial_gradient):
        # TODO implement multiple slopes for single layer
        grads = previous_partial_gradient.copy()
        dfy = self.activation.copy()
        if self.size == 1:
            dfy[dfy > 0] = 0
            self.slope.grad = np.sum(np.sum(grads *dfy))
            grads[self.activation <= 0] *= self.slope.data
        else:
            reshapedactivation = dfy.reshape(-1, dfy.shape[1], order='C')
            reshapedgrads = grads.reshape(-1, grads.shape[1], order='C')
            underzero = reshapedactivation.copy()
            underzero[underzero > 0] = 0
            indexes = reshapedactivation<0
            self.slope.grad = np.sum(np.sum(reshapedgrads * underzero))
            for i,val in enumerate(indexes):
                reshapedgrads[i][val] *= self.slope.data[val]
            grads = np.reshape(reshapedgrads, (np.shape(grads)))
        return grads
