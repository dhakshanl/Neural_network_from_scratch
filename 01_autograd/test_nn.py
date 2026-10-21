"""Tests for nn.py. Run alongside the engine tests:

    python -m pytest -x            everything
    python -m pytest test_nn.py    just these

The last one, test_training_reduces_loss, is the real prize: it actually trains
a network on your engine and checks that the loss goes down. Nothing about
gradients is faked or stubbed - if your backward pass were wrong, it would fail.
"""

import random

from engine import Value
from nn import MLP, Layer, Module, Neuron


# ----------------------------------------------------------------------
# 1. Neuron
# ----------------------------------------------------------------------

def test_neuron_returns_one_value():
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value), f"a Neuron must return one Value, got {type(out)}"


def test_neuron_accepts_raw_floats_and_values():
    """Both work, because your engine wraps bare numbers automatically."""
    n = Neuron(2)
    assert isinstance(n([1.0, 2.0]), Value)
    assert isinstance(n([Value(1.0), Value(2.0)]), Value)


def test_neuron_parameter_count():
    """n weights plus 1 bias."""
    n = Neuron(4)
    assert len(n.parameters()) == 5, f"expected 4 weights + 1 bias, got {len(n.parameters())}"
    assert all(isinstance(p, Value) for p in n.parameters())


def test_neuron_bias_is_included():
    n = Neuron(3)
    assert any(p is n.b for p in n.parameters()), "the bias must be in parameters()"


def test_tanh_neuron_output_is_bounded():
    """With nonlin=True the output is squashed into (-1, 1) however big the input."""
    n = Neuron(1, nonlin=True)
    assert abs(n([1000.0]).data) <= 1.0


def test_linear_neuron_is_not_bounded():
    """With nonlin=False there is no tanh, so the output can be any size."""
    n = Neuron(1, nonlin=False)
    n.w[0].data = 5.0
    assert n([1000.0]).data == 5000.0, "nonlin=False must skip tanh entirely"


# ----------------------------------------------------------------------
# 2. Layer
# ----------------------------------------------------------------------

def test_layer_output_width():
    """Layer(3, 5) maps a list of 3 to a list of 5."""
    layer = Layer(3, 5)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list), f"a Layer returns a list, got {type(out)}"
    assert len(out) == 5, f"expected 5 outputs, got {len(out)}"
    assert all(isinstance(v, Value) for v in out)


def test_layer_parameter_count():
    """5 neurons, each with 3 weights + 1 bias."""
    layer = Layer(3, 5)
    assert len(layer.parameters()) == 5 * 4, f"expected 20, got {len(layer.parameters())}"


def test_layer_neurons_are_different():
    """Random init must break symmetry, or every neuron computes the same thing."""
    layer = Layer(2, 4)
    outputs = [v.data for v in layer([0.7, -0.3])]
    assert len(set(outputs)) > 1, "all neurons produced identical output - weights not random?"


# ----------------------------------------------------------------------
# 3. MLP
# ----------------------------------------------------------------------

def test_mlp_parameter_count():
    """MLP(2, [8, 8, 1]):  8*(2+1) + 8*(8+1) + 1*(8+1) = 24 + 72 + 9 = 105."""
    model = MLP(2, [8, 8, 1])
    assert len(model.parameters()) == 105, f"expected 105, got {len(model.parameters())}"


def test_mlp_single_output_is_unwrapped():
    """A network ending in one neuron returns a Value, not a one-item list."""
    model = MLP(2, [4, 1])
    assert isinstance(model([0.5, -0.5]), Value)


def test_mlp_multi_output_stays_a_list():
    model = MLP(2, [4, 3])
    out = model([0.5, -0.5])
    assert isinstance(out, list) and len(out) == 3


def test_mlp_output_layer_is_linear():
    """The final layer is built with nonlin=False - check it really is."""
    model = MLP(1, [4, 1])
    assert model.layers[-1].neurons[0].nonlin is False
    assert all(n.nonlin for n in model.layers[0].neurons)


# ----------------------------------------------------------------------
# 4. Gradients flow through the whole network
# ----------------------------------------------------------------------

def test_backward_reaches_every_parameter():
    """One backward call must put a gradient on all 105 weights."""
    model = MLP(2, [8, 8, 1])
    loss = (model([0.5, -0.5]) - 1.0) ** 2
    loss.backward()

    zero = [p for p in model.parameters() if p.grad == 0.0]
    assert not zero, f"{len(zero)} parameters got no gradient at all"


def test_zero_grad_clears_everything():
    model = MLP(2, [4, 1])
    loss = (model([0.5, -0.5]) - 1.0) ** 2
    loss.backward()
    assert any(p.grad != 0.0 for p in model.parameters())

    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters()), "zero_grad missed some parameters"


def test_gradients_accumulate_without_zero_grad():
    """Two backwards in a row double the gradient. This is WHY zero_grad exists."""
    model = MLP(2, [4, 1])

    model.zero_grad()
    ((model([0.5, -0.5]) - 1.0) ** 2).backward()
    first = [p.grad for p in model.parameters()]

    ((model([0.5, -0.5]) - 1.0) ** 2).backward()   # no zero_grad in between
    second = [p.grad for p in model.parameters()]

    assert any(abs(b - 2 * a) < 1e-9 and a != 0 for a, b in zip(first, second)), (
        "expected the second backward to add on top of the first"
    )


# ----------------------------------------------------------------------
# 5. The real one: does it actually learn?
# ----------------------------------------------------------------------

def test_training_reduces_loss():
    """Train for 30 steps on the circle dataset and check the loss drops.

    This exercises everything at once: the forward pass, your engine's graph,
    the topological sort, every derivative you wrote, and the update rule.
    """
    rng = random.Random(1)
    data = []
    for _ in range(40):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        data.append(([x, y], 1.0 if x * x + y * y < 1.4 else -1.0))

    model = MLP(2, [8, 8, 1], rng=random.Random(0))

    def total_loss():
        preds = [model(x) for x, _ in data]
        return sum(((p - y) ** 2 for p, (_, y) in zip(preds, data)),
                   start=0) * (1.0 / len(data))

    first = total_loss().data

    for _ in range(30):
        loss = total_loss()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.2 * p.grad

    last = total_loss().data
    assert last < first * 0.5, (
        f"loss went {first:.4f} -> {last:.4f}; training is not working"
    )
