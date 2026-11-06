# Contributing

Vigie is a personal project. Issues and small pull requests are welcome, replies may take time.

## Setup

```bash
uv sync
uv run vigie demo
```

## Before opening a pull request

There is no CI, so please run the checks yourself:

```bash
uv run ruff check .
uv run pytest -q
```

The tests need neither a GPU nor the network.

## Guidelines

- Keep changes small, with a test when the behaviour changes.
- Accuracy claims need a measurement against AIS: see `scripts/validate_ais.py`.
- Never commit a key, a personal path or scan data other than `examples/`.

By contributing, you agree that your work is licensed under the AGPL-3.0.
