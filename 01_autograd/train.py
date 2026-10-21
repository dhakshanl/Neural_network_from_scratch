"""Train the network to separate points inside a circle from points outside.

    python train.py

The dataset is not linearly separable - no straight line divides a disc from
the ring around it. A network with no tanh could never do better than chance
here, which makes it a fair test of whether your nonlinearity actually works.

YOU WRITE the five lines inside the loop. They are the five lines that every
neural network in existence is trained with, from this one up to GPT.
"""

import random

from nn import MLP

LEARNING_RATE = 0.2
STEPS = 80


def make_data(n=100, seed=3):
    """n random points in a square. Inside the circle -> +1, outside -> -1."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.4 else -1.0
        data.append(([x, y], label))
    return data


def main():
    data = make_data()
    model = MLP(2, [8, 8, 1], rng=random.Random(0))

    print(model)
    print(f"{len(model.parameters())} parameters, {len(data)} points\n")

    for step in range(STEPS):
        preds = [model(x) for x, _ in data]
        loss = sum(((p - y) ** 2 for p, (_, y) in zip(preds, data)), 
                   start=0) * (1.0 / len(data))
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= LEARNING_RATE * p.grad
        
        accuracy = sum((p.data > 0) == (y > 0) for p, (_, y) in zip(preds, data))
        if step % 10 == 0 or step == STEPS - 1:
            print(f"step {step:3d}   loss {loss.data:.4f}   "
                  f"accuracy {accuracy / len(data):.0%}")


if __name__ == "__main__":
    main()
