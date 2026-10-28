#!/bin/bash
# Run all open problems through the hosted runner.
# Usage: TASK=../task tools/testall.sh
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/emmy-pycache"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${TASK:-$HERE/../task}"
if [ -z "${PYTHON:-}" ]; then
  for candidate in "$HERE/../venv/bin/python" python3 python; do
    if "$candidate" -c 'import emmy, requests' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python3}"
if [ ! -f "$TASK/submit.py" ]; then
  echo "no submit.py in $TASK; set TASK to the emmy-hackathon-2026 checkout" >&2
  exit 2
fi
cd "$TASK" || exit 1
pass=0
fail=0
failed=()
SERVER_ARGS=()
if [ -n "${RUNNER_URL:-}" ]; then
  SERVER_ARGS=(--server "$RUNNER_URL")
fi
for d in problems/*/; do
  p=$(basename "$d")
  out=$("$PYTHON" submit.py "$p" --compiler "$HERE/compile.py" \
    ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"} 2>&1)
  code=$?
  if [ $code -eq 0 ]; then
    pass=$((pass + 1))
    echo "PASS $p"
  else
    fail=$((fail + 1))
    failed+=("$p")
    echo "FAIL $p"
  fi
  echo "$out" | tail -6 | sed 's/^/    /'
done
echo "---- $pass passed, $fail failed"
if [ "${#failed[@]}" -gt 0 ]; then
  printf '%s\n' "${failed[@]}"
fi
[ "$fail" -eq 0 ]
