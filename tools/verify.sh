#!/usr/bin/env bash
# One-command verification.
#
#   tools/verify.sh              local, no GPU, no network (default)
#   tools/verify.sh --corpora    also run every variant and generated corpus on the local GPU
#   tools/verify.sh --runner     also run all 15 open problems through the hosted runner
#   tools/verify.sh --full       corpora and runner
#
# PYTHON, TASK, COMPILER and VARIANTS are honoured. The exit code is non-zero
# if any executed step fails.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/emmy-pycache"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${PYTHON:-}" ]; then
  for candidate in "$ROOT/../venv/bin/python" python3 python; do
    if "$candidate" -c 'import emmy' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python3}"
TASK="${TASK:-$ROOT/../task}"
COMPILER="${COMPILER:-$ROOT/compile.py}"
RUN_CORPORA=0
RUN_RUNNER=0
CORPORA="stress blocks fixed ln small f16 fuzz attngen attnedge attnadv ops dtypeedge adv"

for arg in "$@"; do
  case "$arg" in
    --quick) RUN_CORPORA=0; RUN_RUNNER=0 ;;
    --corpora) RUN_CORPORA=1 ;;
    --runner) RUN_RUNNER=1 ;;
    --full) RUN_CORPORA=1; RUN_RUNNER=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
  echo "error: $PYTHON is missing or older than 3.12" >&2
  exit 2
fi

pass=0
fail=0
skipped=0

if [ ! -f "$TASK/submit.py" ]; then
  echo "note: no task repository at $TASK; unit tests will skip contract cases" >&2
fi

step() {
  local name="$1"
  shift
  echo "== $name"
  if "$@"; then
    echo "OK $name"
    pass=$((pass + 1))
  else
    echo "FAIL $name"
    fail=$((fail + 1))
  fi
}

skip() {
  echo "SKIP $1: $2"
  skipped=$((skipped + 1))
}

step "dash check" "$PYTHON" "$ROOT/tools/check_dashes.py"
step "Python syntax" "$PYTHON" -m py_compile \
  "$ROOT/compile.py" "$ROOT"/ptxc/*.py "$ROOT"/tools/*.py "$ROOT"/tools/specs/*.py "$ROOT"/tests/*.py
step "unit tests: lowering" "$PYTHON" "$ROOT/tests/test_lowering.py"
step "unit tests: machinery" "$PYTHON" "$ROOT/tests/test_machinery.py"
step "unit tests: contracts" bash -c "\"$PYTHON\" \"$ROOT/tests/test_cli.py\" && \"$PYTHON\" \"$ROOT/tests/test_determinism.py\" && \"$PYTHON\" \"$ROOT/tests/test_no_external.py\""

if [ -f "$TASK/submit.py" ]; then
  step "PTX contract on the open set" "$PYTHON" "$ROOT/tests/check_open_set.py" --task-dir "$TASK"
else
  skip "PTX contract on the open set" "no task repository at $TASK; set TASK"
fi

if [ "$RUN_CORPORA" -eq 1 ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    step "variants (local GPU)" env PYTHON="$PYTHON" COMPILER="$COMPILER" \
      "$ROOT/tools/checkvariants.sh"
    for name in $CORPORA; do
      directory="$ROOT/../$name"
      if [ -d "$directory" ]; then
        step "corpus $name" env PYTHON="$PYTHON" COMPILER="$COMPILER" VARIANTS="$directory" \
          "$ROOT/tools/checkvariants.sh"
      else
        skip "corpus $name" "not generated; run tools/generate_corpus.sh"
      fi
    done
  else
    skip "variants and corpora" "no local NVIDIA device"
  fi
fi

if [ "$RUN_RUNNER" -eq 1 ]; then
  if [ -f "$TASK/submit.py" ]; then
    step "hosted runner (15 open problems)" env PYTHON="$PYTHON" TASK="$TASK" "$ROOT/tools/testall.sh"
  else
    skip "hosted runner" "no task repository at $TASK"
  fi
fi

echo
echo "---- $pass passed, $fail failed, $skipped skipped"
[ "$fail" -eq 0 ]
