#!/usr/bin/env bash
CU=/home/aakash/gemma4/.venv/lib/python3.12/site-packages/nvidia/cu13
echo "=== cu13 layout ==="
ls "$CU" | tr '\n' ' '; echo
echo "=== bin ==="
ls "$CU/bin" 2>/dev/null | head -8 | tr '\n' ' '; echo
echo "=== headers ==="
ls "$CU/include/cuda_runtime.h" 2>/dev/null || echo "no cuda_runtime.h in cu13/include"
find /home/aakash/gemma4/.venv -name cuda_runtime.h 2>/dev/null | head -3
echo "=== nvcc version ==="
"$CU/bin/nvcc" --version 2>&1 | tail -3
echo "=== what flashinfer wants ==="
/home/aakash/gemma4/.venv/bin/python - <<'PY'
import inspect
from flashinfer.jit import cpp_ext
src = inspect.getsource(cpp_ext.get_cuda_path)
print(src)
PY
