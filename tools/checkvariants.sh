#!/bin/bash
# Run the 46-case regression on the local GPU.
# Usage: tools/checkvariants.sh  (override with VARIANTS=..., COMPILER=..., PYTHON=...)
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/emmy-pycache"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VARIANTS="${VARIANTS:-$HERE/tests/variants}"
COMPILER="${COMPILER:-$HERE/compile.py}"
if [ -z "${PYTHON:-}" ]; then
  for candidate in "$HERE/../venv/bin/python" python3 python; do
    if "$candidate" -c 'import emmy, numpy' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python3}"
if [ -f "$VARIANTS/meta.json" ]; then
  cases=("$VARIANTS")
else
  cases=("$VARIANTS"/*/)
fi
if [ ! -e "${cases[0]}" ]; then
  echo "no cases in $VARIANTS" >&2
  exit 2
fi
pass=0; fail=0; failed=()
for d in "${cases[@]}"; do
  p=$(basename "$d")
  out=$("$PYTHON" "$HERE/tools/localrun.py" "$d" "$COMPILER" 2>&1)
  if [ $? -eq 0 ]; then pass=$((pass+1)); echo "PASS $p";
  else fail=$((fail+1)); failed+=("$p"); echo "FAIL $p"; echo "$out" | head -8 | sed 's/^/    /'; fi
done
echo "---- $pass passed, $fail failed"
if [ "${#failed[@]}" -gt 0 ]; then
  printf '%s\n' "${failed[@]}"
fi
[ "$fail" -eq 0 ]
