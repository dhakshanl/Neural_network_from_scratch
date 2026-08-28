# Rung 1 — Scalar Autograd Engine

Build the thing PyTorch's `loss.backward()` is doing, at a scale you can hold in
your head: one number at a time, ~60 lines of your own code.

```
engine.py        the scaffold — you fill in 6 methods
test_engine.py   23 tests. Don't edit this file.
```

## Run

```bash
../.venv/Scripts/activate      # Windows
python -m pytest -x            # -x = stop at the first failure
```

Right now: **23 failed**. That's correct — every method raises
`NotImplementedError`. Your job is to turn that into 23 passed.

## The minimum maths

You do not need a calculus course to start. You need one sentence:

> A derivative tells you how much the output changes when you nudge an input.

If `y = 3x` and you nudge `x` by 0.001, `y` moves by 0.003. So `dy/dx = 3`.
That's it. That's the whole concept.

The second idea is the chain rule, and it's just as small:

> If `a` affects `b`, and `b` affects `c`, then the effect of `a` on `c` is the
> two effects **multiplied**.

Nudge `a` and `b` moves twice as much; nudge `b` and `c` moves five times as
much; so nudging `a` moves `c` ten times as much. `dc/da = dc/db × db/da`.

Backpropagation is that multiplication, applied repeatedly, backwards through a
graph. Nothing else. Every method you're about to write is one local derivative
from the table below, multiplied by whatever gradient arrived from above.

| operation | local derivative |
|---|---|
| `a + b` | 1 for both — addition is a pipe |
| `a * b` | `b` for a, `a` for b — each scaled by the other |
| `a ** n` | `n * a**(n-1)` |
| `relu(a)` | 1 if `a > 0`, else 0 — a gate |
| `tanh(a)` | `1 - tanh(a)**2` — written in terms of the *output* |

## Order to implement

Follow this and the tests unlock in roughly this order:

1. **`__add__`** → `test_add_forward` goes green
2. **`__mul__`** → `test_mul_forward`, `test_scalar_operands`
3. **`backward`** → this is the big one; six tests light up at once
4. **`__pow__`** → and division and subtraction start working for free
5. **`relu`**, **`tanh`**
6. **`test_matches_finite_differences`** — the seven parametrised cases at the
   end. When those pass, your engine is genuinely correct, not just correct on
   the examples I picked.

## The two bugs you will write

Everyone writes at least one of these. Both have a test aimed directly at them.

**1. Assigning instead of accumulating.** `self.grad = ...` instead of
`self.grad += ...`. When a value is used twice (`b = a * a`) gradient arrives
from two directions, and assignment throws one away. → `test_reuse_accumulates`

**2. Calling `_backward()` during the walk.** It's tempting to recurse from the
root calling each node's `_backward` as you reach it. But a node's gradient
isn't final until *every* consumer has deposited into it, and going top-down you
can reach a node down a short path before a longer path has contributed. Build
the complete topological order first, *then* sweep it in reverse. →
`test_topological_order`

## Green? Now break it

Predict the outcome **before** you run each one. Writing your prediction down
first is the part that does the teaching.

| # | Do this | Question to answer |
|---|---|---|
| 1 | Call `.backward()` twice on the same graph | Why do the gradients double? What would you add to fix it? |
| 2 | Delete `self.grad = 1.0` from `backward` | Why is every gradient now exactly 0? |
| 3 | Change `+=` to `=` in `__mul__` | Which tests break, and why *those* ones? |
| 4 | `x = Value(0.0); x.relu().backward()` | What's the gradient at exactly zero? Is that defensible? |
| 5 | Build a chain of 50 `tanh` calls and backward through it | Why is the input's gradient ~0? You just met the **vanishing gradient** — the thing rung 9 is about |
| 6 | `print(len(topo))` for `(a*b+c).tanh()` | Count the nodes by hand first. Did you predict right? |

## You're done with this rung when

You can answer these without looking at the code:

1. Why does `_backward` have to be a *closure* rather than a method?
2. Why must the topological sort happen before any gradient flows?
3. What is `out.grad` at the moment your closure reads it, and who set it?
4. Why does `a / b` get a correct gradient when you never wrote a rule for division?
5. In one sentence, with no jargon: what does `backward()` actually do?

## Next

When all 23 pass, tell me and I'll scaffold the second half: `Neuron`, `Layer`
and `MLP` built on top of your engine, plus a training loop that fits a real
(tiny) dataset. Your `Value` class will run the whole thing without a single
change — which is the moment the design pays off.
