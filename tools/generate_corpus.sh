#!/bin/bash
# Regenerate the local regression corpora with the task's own generator.
#
#   TASK=../task tools/generate_corpus.sh
#   EMMY=tools/emmy-torch tools/generate_corpus.sh
#
# Output defaults to the parent of this repo (../stress, ../blocks, ...), which
# is the layout the harness scripts expect. Set OUT to change it.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${TASK:-$HERE/../task}"
OUT="${OUT:-$HERE/..}"
if [ -z "${EMMY:-}" ]; then
  if [ -x "$HERE/../venv/bin/emmy" ] && "$HERE/../venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
    EMMY="$HERE/../venv/bin/emmy"
  else
    EMMY="$HERE/tools/emmy-torch"
  fi
fi
if [ -z "${PYTHON:-}" ]; then
  for candidate in "$HERE/../venv/bin/python" python3 python; do
    if "$candidate" -c 'import emmy' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python3}"

gen() {
  name="$1"; spec="$2"
  [ -f "$spec" ] || { echo "skip $name (no spec)"; return; }
  echo "== $name"
  "$PYTHON" "$TASK/tools/generate.py" --emmy "$EMMY" --out "$OUT/$name" "$spec" || exit 1
}

gen stress   "$HERE/tools/specs/stress_specs.py"
gen blocks   "$HERE/tools/specs/block_specs.py"
gen fixed    "$HERE/tools/specs/fix_specs.py"
gen ln       "$HERE/tools/specs/ln_specs.py"
gen small    "$HERE/tools/specs/small_specs.py"
gen f16      "$HERE/tools/specs/f16_specs.py"
gen fuzz     "$HERE/tools/specs/fuzz_specs.py"
gen attngen  "$HERE/tools/specs/attn_extra_specs.py"
gen attnedge "$HERE/tools/specs/attn_edge_specs.py"
gen attnadv  "$HERE/tools/specs/attn_adv_specs.py"
gen ops      "$HERE/tools/specs/ops_specs.py"
gen dtypeedge "$HERE/tools/specs/dtype_specs.py"
gen adv      "$HERE/tools/specs/adversarial_specs.py"
