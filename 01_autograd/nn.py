import random
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inputs, nonlin=True, rng=None):
        rng = rng or random.Random(0)
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.nonlin = nonlin      

    def __call__(self, x):
        acc = self.b
        pairs = zip(self.w, x)
        for w, i in pairs:
            acc += w * i
        return acc.tanh() if self.nonlin else acc
            
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = "tanh" if self.nonlin else "linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    """A row of neurons, every one of them fed the same input."""

    def __init__(self, n_inputs, n_outputs, rng=None, **kwargs):
        rng = rng or random.Random(0)
        self.neurons = [Neuron(n_inputs, rng=rng, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
        
    def __repr__(self):
        return f"Layer[{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, n_inputs, layer_sizes, rng=None):
        rng = rng or random.Random(0)
        sizes = [n_inputs] + list(layer_sizes)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i < len(layer_sizes) - 1), rng=rng)
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
