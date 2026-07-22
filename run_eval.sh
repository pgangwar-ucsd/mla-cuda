#!/bin/bash
#
# Usage examples (args are forwarded to exhaustive_benchmark_suite.py):
#   bash run_eval.sh --quick
#   bash run_eval.sh --cuda-attn-only --decode-focus
#   bash run_eval.sh --cuda-attn-only --scenarios decode_single_64k_cache decode_production_avg_cache
#   bash run_eval.sh --scenario decode_production_avg_cache --cuda-attn-only
#
# Skip rebuild: MLA_SKIP_BUILD=1 bash run_eval.sh --decode-focus --cuda-attn-only

CONDA_ENV="${MLA_CONDA_ENV:-ece285}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
if [ -d "$CONDA_ROOT/envs/$CONDA_ENV/bin" ]; then
  export PATH="$CONDA_ROOT/envs/$CONDA_ENV/bin:$PATH"
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: PyTorch not found. Activate a CUDA env or set MLA_CONDA_ENV."
  exit 1
fi

echo ""
echo " 1. CLEANING OLD BUILDS                   "
echo ""

if [ "${MLA_SKIP_BUILD:-0}" = "1" ]; then
  echo "  (MLA_SKIP_BUILD=1 — skipping clean, compile, and pip install)"
else
pip uninstall -y mla_custom_cuda || true
rm -rf build/
rm -rf mla_custom_cuda.egg-info/

echo ""
echo " 2. COMPILING CUSTOM CUDA KERNEL          "
echo ""

#We install locally. The JIT compiler will output errors here if your C++ syntax is wrong.
set -e
pip install -e .

fi

echo ""
echo " 3. RUNNING EXHAUSTIVE TEST SUITE         "
echo ""
set -e
python exhaustive_benchmark_suite.py "$@"