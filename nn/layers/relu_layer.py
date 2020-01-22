import numpy as np
from numba import njit, prange, jit
from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.activation = []
    def forward(self, data):
        activation = data.copy()
        self.activation = activation
        activation = np.maximum(activation,0)
        return activation


    def backward(self, previous_partial_gradient):
        grads = previous_partial_gradient.copy()
        grads[self.activation <= 0] = 0
        return grads



class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None


    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
       finalval = []
       for i in data:
           if i < 0:
               finalval.append(0)
           else:
               finalval.append(i)

       return finalval

        # TODO Helper function for computing ReLU

    def forward(self, data):
        activation = data.copy()
        vectorizeddata = np.ndarray.flatten(activation)
        output = self.forward_numba(vectorizeddata)
        self.data = output # store before reshaping
        output = np.reshape(output, (np.shape(data)))

        return output


    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        finalgradvals = []
        for ind, i in enumerate(data):
           if i <= 0:
               finalgradvals.append(0)
           else:
               finalgradvals.append(grad[ind])
        return finalgradvals


    def backward(self, previous_partial_gradient):
        # TODO
        pgrad = previous_partial_gradient.copy()
        vectorizedpgrad = np.ndarray.flatten(pgrad)
        gradvals = self.backward_numba(self.data, vectorizedpgrad)
        gradvals = np.reshape(gradvals,np.shape(previous_partial_gradient))
        return gradvals
