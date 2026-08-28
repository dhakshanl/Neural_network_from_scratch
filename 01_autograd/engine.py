"""A scalar automatic differentiation engine. YOU implement the marked methods.

THE IDEA
--------
When you write `d = a * b + c`, Python computes a number and throws away how it
got there. We are going to keep that history.

Every `Value` holds a number AND a pointer back to the Values that produced it.
So an expression builds a graph as a side effect of being evaluated:

    a --+
        +--(*)--> e --+
    b --+             +--(+)--> d
    c ----------------+

Then `d.backward()` walks that graph in reverse and fills in `.grad` on every
node: "if I nudge this number up a tiny bit, how much does d change?"

THE TWO-PHASE TRICK
-------------------
Each operation does two things:

  1. FORWARD (runs immediately): compute the output number, and record the
     children so the graph exists.
  2. BACKWARD (runs later): a closure stored on the output node that knows the
     local derivative of THIS one operation, and pushes gradient into the
     children.

A node never needs to know anything about the rest of the graph. It only knows
how to turn "gradient on my output" into "gradient on my inputs". Chaining
those together IS the chain rule, and it is the entire content of backprop.

TWO RULES THAT WILL BITE YOU
----------------------------
  * ACCUMULATE with `+=`, never assign with `=`. If a Value is used twice
    (`b = a * a`), gradient arrives from two directions and both count.
    Assignment throws one away. test_reuse_accumulates checks this.

  * `out.grad` is not known when the closure is DEFINED, only when it is CALLED.
    Write `self.grad += <local derivative> * out.grad` inside the function body
    so it reads the value that exists at call time.

ORDER TO IMPLEMENT (the tests unlock roughly in this order)
    __add__  ->  __mul__  ->  backward  ->  __pow__  ->  relu  ->  tanh

Run `python -m pytest -x` after each one. `-x` stops at the first failure, so
you always have exactly one thing to fix.
"""

import math


class Value:
    """A single scalar number, and the operation that produced it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)

        # dL/d(self), where L is whatever node you eventually call .backward()
        # on. Starts at zero: "so far, nothing tells me this number matters."
        self.grad = 0.0

        # Filled in by whichever operation created this node. A leaf (a number
        # you typed) has no producing operation, so its backward does nothing.
        self._backward = lambda: None

        # The Values this one was built from, in argument order. A tuple
        # rather than a set so the graph walk is deterministic - two runs
        # of the same program visit nodes in the same order.
        self._prev = tuple(_children)
        self._op = _op                # label, useful when printing the graph

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------
    # The operations you implement
    # ------------------------------------------------------------------

    def __add__(self, other):
        """self + other

        FORWARD:  out.data = self.data + other.data

        LOCAL DERIVATIVE:  d(a+b)/da = 1  and  d(a+b)/db = 1
        Addition is a pipe. Whatever gradient arrives at the output passes
        through to BOTH inputs unchanged.

        So _backward should do:   self.grad  += 1.0 * out.grad
                                  other.grad += 1.0 * out.grad

        Steps:
          1. other = _wrap(other)          # so `x + 3` works
          2. out = Value(<data>, (self, other), "+")
          3. define  def _backward(): ...  # accumulate into both children
          4. out._backward = _backward
          5. return out
        """
        raise NotImplementedError("implement Value.__add__")

    def __mul__(self, other):
        """self * other

        FORWARD:  out.data = self.data * other.data

        LOCAL DERIVATIVE:  d(a*b)/da = b  and  d(a*b)/db = a
        Each input's gradient is scaled by THE OTHER input's value. Intuition:
        if b is 10, nudging a by 0.001 moves the product by 0.01.

        So:  self.grad  += other.data * out.grad
             other.grad += self.data  * out.grad

        Same five steps as __add__, with "*" as the op label.
        """
        raise NotImplementedError("implement Value.__mul__")

    def __pow__(self, exponent):
        """self ** exponent, where exponent is a plain int or float.

        FORWARD:  out.data = self.data ** exponent

        LOCAL DERIVATIVE:  d(a**n)/da = n * a**(n-1)     (the power rule)

        So:  self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        There is no `other.grad` here: the exponent is a constant, not a Value,
        so there is nothing to send gradient back into. Keep it that way -
        supporting a Value exponent needs logarithms and is not today's point.

        Why bother with pow at all? Because this one method gives you division
        for free: a / b is just a * b**-1. See __truediv__ below.
        """
        assert isinstance(exponent, (int, float)), "only int/float exponents"
        raise NotImplementedError("implement Value.__pow__")

    def relu(self):
        """max(self, 0)

        FORWARD:  out.data = self.data if self.data > 0 else 0.0

        LOCAL DERIVATIVE:  1 if the input was positive, else 0.
        ReLU is a gate: gradient flows through where the unit was active and is
        blocked where it was not.

        So:  self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        """
        raise NotImplementedError("implement Value.relu")

    def tanh(self):
        """Hyperbolic tangent, squashing any input into (-1, 1).

        FORWARD:  t = math.tanh(self.data)

        LOCAL DERIVATIVE:  d(tanh(a))/da = 1 - tanh(a)**2

        This is the derivative worth noticing: it is written in terms of the
        OUTPUT, not the input. You already computed t in the forward pass, so
        backward costs one multiply. Cache the forward result, reuse it in
        backward - that pattern is everywhere in real frameworks.

        So:  self.grad += (1 - t**2) * out.grad
        """
        raise NotImplementedError("implement Value.tanh")

    def backward(self):
        """Fill in .grad on every node in the graph that produced self.

        Two parts.

        PART 1 - ORDER. You cannot call a node's _backward() until every node
        that CONSUMES it has already run and deposited gradient into it.
        Otherwise you propagate a half-finished number onwards. The correct
        order is a reverse topological sort of the graph.

        Build it with a depth-first search:

            topo, visited = [], set()

            def build(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build(child)
                    topo.append(v)      # appended AFTER all of its children

            build(self)

        Now `topo` holds children before parents, so `reversed(topo)` gives you
        parents before children - exactly what you need.

        PART 2 - SEED AND SWEEP.

            self.grad = 1.0             # d(self)/d(self) = 1. Backprop needs
                                        # somewhere to start; this is it.
            for node in reversed(topo):
                node._backward()

        That is the whole algorithm. About fifteen lines.
        """
        raise NotImplementedError("implement Value.backward")

    # ------------------------------------------------------------------
    # Free, once the above work. Note how few primitives you actually needed:
    # negation, subtraction and division are all add / mul / pow in disguise,
    # so they get correct gradients without any new derivative rules.
    # ------------------------------------------------------------------

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
    """Lets you write `x + 3` instead of `x + Value(3)`."""
    return x if isinstance(x, Value) else Value(x)
