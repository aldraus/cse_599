class BaseOptimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        pass

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
