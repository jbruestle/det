#!/bin/sh
# Builds the CUDA library and runs every bit-exactness check (kernel suites + end-to-end on the 4B dev model).
set -e
cd "$(dirname "$0")"
make -C cuda libdet.so
PY=.venv/bin/python
for t in glue gemm attn gdn sample; do echo "== tests/test_$t.py"; $PY tests/test_$t.py | tail -1; done
if [ -d models/qwen35-4b-w6 ]; then
  echo "== end-to-end (4B)"; $PY tools/cuda_compare.py models/qwen35-4b-w6 out/hf_4b_baseline.npz 3 | tail -1
  $PY tools/invariance.py models/qwen35-4b-w6 out/ppl_hf_4b.npz | tail -1
fi
