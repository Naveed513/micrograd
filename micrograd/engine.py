import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def _value(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._value(other)
        out = Value(self.data+other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self._value(other)
        out = Value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Support only int/float power"
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def exp(self):
        e = math.exp(round(self.data, 4))
        out = Value(e, (self, ), 'exp')
        def _backward():
            self.grad = e * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        e = self.exp()
        t = (e - 1) / (e + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        r = self.data if self.data > 0 else 0
        out = Value(r, (self, ), 'ReLU')
        def _backward():
            self.grad = (r > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def topo_builder(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_builder(child)
                topo.append(v)
        topo_builder(self)
        self.grad = 1.0
        for child in reversed(topo):
            child._backward()

    def __repr__(self):
        return f"Value(data={self.data}, bias={self.grad})"
    