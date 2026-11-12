#!/usr/bin/env bash
V=/home/aakash/gemma4/.venv
echo "=== torch CUDA ==="
"$V/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
echo "=== flashinfer expects ==="
"$V/bin/python" -c "import flashinfer; print('flashinfer', flashinfer.__version__)" 2>&1 | tail -1
echo "=== cu12 nvcc present? ==="
ls -d "$V/lib/python3.12/site-packages/nvidia"/cuda_nvcc* 2>/dev/null || echo "no cuda_nvcc (cu12) package"
find "$V" -path "*cuda_nvcc*" -name nvcc -type f 2>/dev/null | head -2
echo "=== cuda.h version in cu13 headers ==="
grep -m1 "define CUDA_VERSION" "$V/lib/python3.12/site-packages/nvidia/cu13/include/cuda.h" 2>/dev/null || echo "(no cuda.h)"
echo "=== triton's bundled cuda ==="
ls "$V/lib/python3.12/site-packages/triton/backends/nvidia/bin" 2>/dev/null | tr '\n' ' '; echo
grep -m1 "define CUDA_VERSION" "$V/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda.h" 2>/dev/null || echo "(no triton cuda.h)"
