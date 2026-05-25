#!/usr/bin/env bash
# Pre-publish gate: TypeScript + Python checks and smoke imports.
# Run from repo root: ./scripts/pre_release_check.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve a Python >= 3.10 interpreter once. Default to `python3` on Unix
# but fall back to `python` if it's the only one installed.
if command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "FATAL: neither 'python3' nor 'python' found on PATH" >&2
  exit 2
fi

echo "==> Bun: install, typecheck, test"
bun install --frozen-lockfile
bun run typecheck
bun test

echo "==> Python: install, lint, format, test (${PYTHON})"
"${PYTHON}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" \
  || { echo "FATAL: ${PYTHON} is older than 3.10"; exit 1; }
"${PYTHON}" -m pip install -e ".[dev]"
"${PYTHON}" -m ruff check packages tests scripts apps/viewer
"${PYTHON}" -m black --check packages tests scripts apps/viewer
"${PYTHON}" -m pytest -q

echo "==> Python: import smoke test"
"${PYTHON}" -c "import stream; import proactivity; import reflections; from proactivity.agent import ProactivityAgent"

echo "==> Viewer: --help-only"
"${PYTHON}" -m apps.viewer --help-only

echo "Pre-release check passed."
