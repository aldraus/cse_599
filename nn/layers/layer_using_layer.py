from abc import ABC

import numpy as np

from .layer import Layer
from .dummy_layer import DummyLayer


class LayerUsingLayer(Layer, ABC):
    def __init__(self, parent=None):
        if parent is None:
            parent = DummyLayer()
        super(LayerUsingLayer, self).__init__(parent)

    @property
    def final_layer(self):
        raise NotImplementedError

    def set_parent(self, val):
        if isinstance(self._parent, DummyLayer):
            self._parent.set_parent(val)
        else:
            self._parent = val

    def backward(self, previous_partial_gradients=None) -> np.ndarray:
        if previous_partial_gradients is not None:
            gradient = self.final_layer.backward(previous_partial_gradients)
        else:
            gradient = self.final_layer.backward()

        # Create graph
        frontier = [self.final_layer]
        graph = {}
        while len(frontier) > 0:
            node = frontier.pop()
            if node.parents is None:
                continue
            for parent in node.parents:
                if parent not in graph:
                    graph[parent] = set()
                graph[parent].add(node)
                frontier.append(parent)

        # Topological sort
        order = []
        frontier = [self.final_layer]
        while len(frontier) > 0:
            node = frontier.pop()
            order.append(node)
            if node.parents is None:
                continue
            for parent in node.parents:
                graph[parent].remove(node)
                if len(graph[parent]) == 0:
                    frontier.append(parent)

        grad_dict = {}
        self._assign_parent_grads(self.final_layer, gradient, grad_dict)
        # Ignore loss layer because already computed
        order = order[1:]
        # Send gradients backwards
        for layer in order:
            output_grad = layer.backward(grad_dict[layer])
            if layer.parents is not None:
                self._assign_parent_grads(layer, output_grad, grad_dict)
        return output_grad

    @staticmethod
    def _assign_parent_grads(layer, grad, grad_dict):
        assert isinstance(grad, np.ndarray) or isinstance(grad, tuple), (
            "grad should be a nparray or a tuple of nparrays but was %s." % type(grad).__name__)
        assert isinstance(layer.parent, tuple) == isinstance(grad, tuple), (
            "Gradients should be a tuple iff there are multiple parents.")
        if not isinstance(grad, tuple):
            grad = (grad,)
        for parent, grad in zip(layer.parents, grad):
            if parent in grad_dict:
                grad_dict[parent] = grad_dict[parent] + grad
            else:
                grad_dict[parent] = grad
