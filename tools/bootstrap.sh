#!/usr/bin/env bash
# Prepare a Linux or WSL2 machine for this repository.
#
#   tools/bootstrap.sh              print the commands
#   tools/bootstrap.sh --run        execute them
#   tools/bootstrap.sh --run --with-torch
#
# Defaults place the task repository and the virtual environment next to this
# checkout: ../task and ../venv. Override with DEST, TASK or PYTHON.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-$(dirname "$ROOT")}"
TASK_DIR="${TASK:-$DEST/task}"
VENV="${VENV:-$DEST/venv}"
PYTHON="${PYTHON:-python3}"
EXECUTE=0
WITH_TORCH=0

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --run) EXECUTE=1 ;;
    --with-torch) WITH_TORCH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; install Python 3.12 or set PYTHON" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "error: $PYTHON is older than 3.12" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git not found" >&2
  exit 1
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$EXECUTE" -eq 1 ]; then
    "$@"
  fi
}

echo "repository: $ROOT"
echo "destination: $DEST"
echo "task: $TASK_DIR"
echo "venv: $VENV"
echo

if [ ! -d "$DEST" ]; then
  run mkdir -p "$DEST"
fi

if [ -x "$VENV/bin/python" ]; then
  echo "venv already exists, reusing it"
else
  run "$PYTHON" -m venv "$VENV"
fi

run "$VENV/bin/python" -m pip install --upgrade pip
run "$VENV/bin/python" -m pip install -r "$ROOT/requirements-tools.txt"

if [ -f "$TASK_DIR/submit.py" ]; then
  echo "task repository already exists, leaving it alone"
elif [ -e "$TASK_DIR" ]; then
  echo "error: $TASK_DIR exists but has no submit.py" >&2
  exit 1
else
  run git clone https://github.com/cloudrift-ai/emmy-hackathon-2026.git "$TASK_DIR"
fi

if [ "$WITH_TORCH" -eq 1 ]; then
  run "$VENV/bin/python" -m pip install torch
fi

echo
echo "next:"
echo "  cd $ROOT"
echo "  $VENV/bin/python tools/doctor.py --runner"
echo "  PYTHON=$VENV/bin/python tools/verify.sh --quick"
if [ "$EXECUTE" -eq 0 ]; then
  echo
  echo "nothing was executed; add --run to apply the commands"
fi
