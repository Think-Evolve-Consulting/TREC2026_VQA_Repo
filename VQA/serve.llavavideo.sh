#!/bin/bash
# Serve LLaVA-Video-7B (bf16) with vLLM 0.12 from the Cosmos venv on port 8009,
# for VQA/rank.options.mc.llavavideo.py. Runs in the foreground: the WSL2 distro
# lives only as long as this session does.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/serve.llavavideo.sh
#
# Measured on the 3090: 15.03 GiB of weights, 5.61 GiB of KV cache (105,104
# tokens), ~6,570 prompt tokens per call at 32 frames.
systemctl --user stop cosmos-reason2 2>/dev/null
exec /home/aakash/cosmos-reason2/.venv/bin/vllm serve Isotr0py/LLaVA-Video-7B-Qwen2-hf \
    --host 0.0.0.0 --port 8009 --served-model-name llava-video-7b \
    --allowed-local-media-path /mnt/e/TREC_2026 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
    --limit-mm-per-prompt '{"video": 1}' \
    --media-io-kwargs '{"video": {"num_frames": 32}}' \
    --skip-mm-profiling \
    > /mnt/e/TREC_2026/logs/llava.server.log 2>&1
