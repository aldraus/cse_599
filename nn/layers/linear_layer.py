from typing import Optional, Callable

import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    hidden_layer_output = np.array([])
    def __init__(self, input_size: int, output_size: int, parent=None):
        super(LinearLayer, self).__init__(parent)
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))

        # TODO create the weight parameter
        self.weight = Parameter(np.zeros((input_size, output_size)), dtype=np.float32)
        self.initialize()

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Linear conv_layers (fully connected) forward pass
        :param data: n X d array (batch x features)
        :return: n X c array (batch x channels)
        """

        self.hidden_layer_output = data
        output = np.dot(data, self.weight.data) + self.bias.data
        return output

        # TODO do the linear conv_layers
        return None


    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future conv_layers
        :return: gradients wrt inputs
        """
        # TODO do the backward step
        grad_bias = np.sum(previous_partial_gradient, axis=0, keepdims=True)
        grad_weight = np.dot(self.hidden_layer_output.T, previous_partial_gradient)
        self.bias.grad = grad_bias
        self.weight.grad = grad_weight
        grads = np.dot(previous_partial_gradient, self.weight.data.T)

        return grads

    def selfstr(self):
        return str(self.weight.data.shape)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(LinearLayer, self).initialize()
