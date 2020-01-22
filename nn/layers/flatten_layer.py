from .layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.data = np.array([])
    def forward(self, data):
        self.data = data
        # TODO reshape the data here and return it (this can be in place).
        return np.reshape(data, (data.shape[0], -1))

    def backward(self, previous_partial_gradient):
        # TODO
        return np.reshape(previous_partial_gradient,(self.data.shape))
