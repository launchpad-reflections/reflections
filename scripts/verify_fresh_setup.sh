#!/usr/bin/env bash
# Fresh-clone sanity check — no glasses, MediaMTX, or ngrok required.
# Run from anywhere: ./scripts/verify_fresh_setup.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve a Python >= 3.10 interpreter once. Some Linux distros and many
# macOS shells leave `python` pointing at 2.x or 3.9 even when Python 3.10+
# is installed as `python3`.
if command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "FATAL: neither 'python3' nor 'python' found on PATH" >&2
  exit 2
fi

declare -a STEP_NAMES=()
declare -a STEP_RESULTS=()
FAILED=0

run_step() {
  local name="$1"
  shift
  echo ""
  echo "==> ${name}"
  STEP_NAMES+=("${name}")
  if "$@"; then
    STEP_RESULTS+=("PASS")
    echo "    PASS"
  else
    STEP_RESULTS+=("FAIL")
    FAILED=$((FAILED + 1))
    echo "    FAIL" >&2
  fi
}

echo "Reflections fresh-clone verify"
echo "  repo: ${REPO_ROOT}"

run_step "Python >= 3.10 (${PYTHON})" \
  "${PYTHON}" -c "import sys; v = sys.version_info; print(f'    {sys.executable} ({sys.version.split()[0]})'); raise SystemExit(0 if v >= (3, 10) else 1)"

run_step 'pip install -e ".[dev]"' \
  "${PYTHON}" -m pip install -e ".[dev]"

run_step "pytest -q" \
  "${PYTHON}" -m pytest -q

run_step "bun install --frozen-lockfile" \
  bun install --frozen-lockfile

run_step "bun run typecheck" \
  bun run typecheck

run_step "bun test" \
  bun test

run_step "python -m apps.viewer --help-only" \
  "${PYTHON}" -m apps.viewer --help-only

run_step "import stream, proactivity, reflections, ProactivityAgent" \
  "${PYTHON}" -c "
import stream
import proactivity
import reflections
from proactivity.agent import ProactivityAgent
print('imports ok')
"

echo ""
echo "========== Summary =========="
for i in "${!STEP_NAMES[@]}"; do
  printf "  [%s] %s\n" "${STEP_RESULTS[$i]}" "${STEP_NAMES[$i]}"
done
echo "============================="

if [[ "${FAILED}" -gt 0 ]]; then
  echo ""
  echo "${FAILED} check(s) failed." >&2
  exit 1
fi

echo ""
echo "All checks passed."
exit 0
