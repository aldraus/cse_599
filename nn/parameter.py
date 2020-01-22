import numpy as np


class Parameter(object):
    def __init__(self, data: np.ndarray, dtype=np.float32):
        self._data = data.astype(dtype)
        self._grad = np.zeros_like(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data[:] = data

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad[:] += grad

    def zero_grad(self):
        self._grad[:] = 0
