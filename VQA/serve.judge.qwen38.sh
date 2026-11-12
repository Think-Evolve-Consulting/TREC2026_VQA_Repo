#!/bin/bash
# Serve Qwen3.8-27B INT4 for VQA/judge.rank.mc.py on port 8011.
# The judge sends frames as IMAGES, not video, so --limit-mm-per-prompt must
# allow 16 of them -- and doing it this way side-steps the do_sample_frames
# trap that applies only to this model's native video path.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/serve.judge.qwen38.sh
#
# vLLM 0.27 is required (only 0.27 registers Qwen3_5ForConditionalGeneration);
# both env vars are mandatory under WSL2 (UVA, and FlashInfer's JIT sampler).
systemctl --user stop cosmos-reason2 2>/dev/null
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
# Without these the server loads, answers /v1/models, passes preflight, then
# EngineDeadErrors on the FIRST real request: FlashInfer JIT-builds its
# batch-prefill module then and looks for nvcc via CUDA_HOME, CUDA_PATH, PATH,
# /usr/local/cuda -- none of which resolve in a bare shell. The toolkit ships
# inside the venv. Copied from ~/.config/systemd/user/qwen38count-vllm.service.
CU13=/home/aakash/gemma4/.venv/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CU13"
export CUDA_PATH="$CU13"
# The venv bin must be on PATH too: FlashInfer shells out to ninja
# to build that module, and ninja is installed as a venv package.
export PATH="$CU13/bin:/home/aakash/gemma4/.venv/bin:$PATH"
exec /home/aakash/gemma4/.venv/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \
    --host 0.0.0.0 --port 8011 \
    --served-model-name RedHatAI/Qwen3.8-27B-INT4 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
    --limit-mm-per-prompt '{"image": 16}' \
    --skip-mm-profiling     --attention-backend TRITON_ATTN --kv-cache-dtype bfloat16 \
    > /mnt/e/TREC_2026/logs/judge.qwen38.server.log 2>&1
