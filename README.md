# ai_projects

Working through deep learning from scratch — one numbered folder per project.

## Setup

```bash
.venv/Scripts/activate        # Windows
python -m pip install -r requirements.txt
```

Shared `.venv` for every project. Datasets live in each project's `data/` and are
gitignored — they're downloadable, they don't belong in history.

## Projects

| # | Project | Status |
|---|---------|--------|
| 01 | Neural network from scratch (MNIST MLP, numpy) | not started |

Add a row when you start one. Each project folder gets its own `README.md`
explaining the math, not just how to run it.

## The ladder

The full beginner → cracked progression, with what makes each rung harder than
the last: **[The Cracked Ladder](https://claude.ai/artifact-url-pending)**

## Rules

1. No frameworks for the thing the project is meant to teach.
2. Gradient-check before trusting a backward pass.
3. Predict the outcome before running the experiment.
