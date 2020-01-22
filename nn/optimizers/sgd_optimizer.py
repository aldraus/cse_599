from .base_optimizer import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate):
        super(SGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - self.learning_rate * parameter.grad
