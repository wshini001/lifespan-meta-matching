# Development

Run these commands from the repository root after installing `requirements.txt`.

## Format and lint

Check formatting and static code rules:

```bash
python -m black --check .
python -m ruff check .
```

Apply formatting:

```bash
python -m black .
```

Formatting and lint settings are defined in `pyproject.toml`.
