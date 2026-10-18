#!/bin/sh
# Builds the Metal library and runs every bit-exactness check (primitives, kernel suites, end-to-end on the 4B dev model).
set -e
cd "$(dirname "$0")"
make -C metal libdet_metal.dylib
PY=.venv/bin/python
for t in prim denorm glue attn gdn sample; do echo "== tests/metal/test_$t.py"; $PY tests/metal/test_$t.py | tail -1; done
echo "== tests/metal/test_gemm.py quick"; $PY tests/metal/test_gemm.py quick | tail -1
if [ -d models/qwen35-4b-w6 ]; then
  echo "== end-to-end (4B)"; $PY tools/metal_compare.py models/qwen35-4b-w6 ${HF_BASELINE:-dl/qwen35-4b} 3 --chunk 5 | tail -1
fi
