# Contributing

Thank you for your interest in contributing to Reflections!

## Getting started

1. Fork the repository and clone your fork.
2. Follow [docs/SETUP.md](docs/SETUP.md) for local development.
3. Install the Python package with dev extras: `pip install -e ".[dev]"`
4. Install Bun dependencies: `bun install`
5. (Recommended) Install the pre-commit hooks: `pre-commit install`
6. Create a branch for your change: `git checkout -b feat/my-change`

## Development workflow

| Component | Entrypoint |
|-----------|------------|
| Camera server | `bun run dev` |
| Viewer | `python -m apps.viewer` |
| Proactivity dashboard | `python -m proactivity.dashboard` |

## Tests and checks

Before opening a PR, run the same checks CI runs:

```bash
# TypeScript
bun run typecheck
bun test

# Python
ruff check packages tests scripts apps/viewer
black --check packages tests scripts apps/viewer
pytest -q
```

Or run everything at once with the gate script:

```bash
./scripts/pre_release_check.sh         # macOS / Linux
.\scripts\pre_release_check.ps1        # Windows PowerShell
```

## Code style

- **Python**: follow existing conventions in `packages/`. Ruff and Black config live in `pyproject.toml` (`line-length = 100`).
- **TypeScript**: match patterns in `apps/camera-server/`.
- Keep diffs focused — avoid unrelated refactors in the same PR.

## Pull requests

1. Update documentation if you change env vars, entrypoints, or architecture.
2. Do not commit secrets (`.env`, API keys, OAuth tokens).
3. Do not commit model weights, face galleries, or runtime logs.
4. Fill out the PR template with a test plan.

## Reporting issues

Use the GitHub issue templates for bugs and feature requests. Include OS, Python/Bun versions, and steps to reproduce.

## Agent contributors

See [AGENTS.md](AGENTS.md) for architecture reference and key file paths.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
