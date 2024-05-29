from micrograd.engine import Value
import random

class Neuron:
    def __init__(self, nin, nonlin=None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin is not None:
            if self.nonlin == 'tanh':
                return act.tanh()
            elif self.nonlin == 'relu':
                return act.relu()
            else:
                raise NotImplementedError
        return act
    
    def parameters(self):
        return f"Neuron(self.w={self.w}, self.b={self.b})"

class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts, act='relu'):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=act if i != len(nouts)-1 else None)
              for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

