import math
class Value:
    def __init__(self, data, _children = (), _op = ""):
        self.data = float(data)
        self._op = _op
        self._prev = tuple(_children)
        self._backward = lambda: None
        self.grad = 0.0


    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other=_wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other=_wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
     
    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data>0 else 0.0) * out.grad 
        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float exponents"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")
        def _backward():
            self.grad += exponent*((self.data)**(exponent-1))*out.grad
        out._backward=_backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(s):
            if s not in visited:
                visited.add(s)
                for child in s._prev:
                    build(child)
                topo.append(s)
        build(self)
        self.grad = 1.0
        for value in reversed(topo):
            value._backward()
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-_wrap(other))

    def __truediv__(self, other):
        return self * _wrap(other) ** -1

    def __radd__(self, other):      # 3 + x
        return self + other

    def __rmul__(self, other):      # 3 * x
        return self * other

    def __rsub__(self, other):      # 3 - x
        return _wrap(other) + (-self)

    def __rtruediv__(self, other):  # 3 / x
        return _wrap(other) * self ** -1


def _wrap(x):
    return x if isinstance(x, Value) else Value(x)