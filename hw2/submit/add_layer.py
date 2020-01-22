from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .layer import Layer


class AddLayer(Layer):
    def __init__(self, parents):
        super(AddLayer, self).__init__(parents)
        self.numinput = 0
    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        # expect all tensors to have same shape
        self.numinput = len(inputs)
        addedtensors = sum(inputs)

        return addedtensors

    def backward(self, previous_partial_gradient) -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        return(previous_partial_gradient,) * self.numinput # TODO just returning two exact same versions ?


