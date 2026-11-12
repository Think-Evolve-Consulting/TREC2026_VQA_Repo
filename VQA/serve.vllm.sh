#!/bin/bash
# Serve one of this project's vLLM models in the foreground, for
# VQA/bench.tokens.py and for the ranking passes.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/serve.vllm.sh qwen3vl
#
# Foreground on purpose: WSL2 tears the whole distro down when its last session
# exits, so a backgrounded server dies with the window that started it.
#
# One model fits on the 3090 at a time. cosmos-reason2 is a systemd user
# service that starts with the distro and takes ~16 GB, so every case stops it
# first. The two vLLM 0.27 models (gemma4, qwen38) need the env vars in
# VLLM_027_ENV: the V2 model runner wants UVA, which WSL2's GPU-PV does not
# expose, and the FlashInfer sampler JIT-compiles against a CUDA toolkit that
# is not installed here -- it fires at the first sampled token, so the server
# looks healthy until a request arrives.
set -u

MODEL="${1:-}"
COSMOS_VENV=/home/aakash/cosmos-reason2/.venv
GEMMA_VENV=/home/aakash/gemma4/.venv
MEDIA=/mnt/e/TREC_2026
LOGS=$MEDIA/logs

mkdir -p "$LOGS"
systemctl --user stop cosmos-reason2 2>/dev/null
pkill -f "[v]llm serve" 2>/dev/null
sleep 3

case "$MODEL" in
cosmos)
    exec $COSMOS_VENV/bin/vllm serve nvidia/Cosmos-Reason2-2B \
        --host 0.0.0.0 --port 8008 \
        --allowed-local-media-path $MEDIA \
        --max-model-len 24576 --gpu-memory-utilization 0.75 --enforce-eager \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --reasoning-parser qwen3 \
        > "$LOGS/bench.server.cosmos.log" 2>&1
    ;;
qwen3vl)
    exec $COSMOS_VENV/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct \
        --host 0.0.0.0 --port 8009 \
        --served-model-name Qwen/Qwen3-VL-8B-Instruct \
        --allowed-local-media-path $MEDIA \
        --max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --skip-mm-profiling \
        > "$LOGS/bench.server.qwen3vl.log" 2>&1
    ;;
llavavideo)
    exec bash "$(dirname "$0")/serve.llavavideo.sh"
    ;;
gemma4)
    export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
    exec $GEMMA_VENV/bin/vllm serve google/gemma-4-12B-it-qat-w4a16-ct \
        --host 0.0.0.0 --port 8010 \
        --served-model-name google/gemma-4-12B-it-qat-w4a16-ct \
        --allowed-local-media-path $MEDIA \
        --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
        --limit-mm-per-prompt '{"image": 32}' \
        --reasoning-parser gemma4 \
        > "$LOGS/bench.server.gemma4.log" 2>&1
    ;;
qwen38)
    export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
    # --attention-backend is a third thing this model needs, beyond the two
    # env vars every 0.27 service here carries. Left to choose, vLLM 0.27.1
    # picks FlashInfer for Qwen3.8's hybrid attention ("Using FLASHINFER
    # attention backend out of potential backends: ['FLASHINFER',
    # 'TRITON_ATTN']"), and FlashInfer's prefill wrapper JIT-compiles CUDA the
    # first time it plans an attention call. There is no CUDA toolkit here, so
    # the engine dies on the first request with "Could not find nvcc" and takes
    # the server with it -- every later request gets connection-refused. That
    # is a different trap from the FlashInfer *sampler*, which
    # VLLM_USE_FLASHINFER_SAMPLER=0 already covers (the log confirms it is
    # disabled) and which is not what fires here. TRITON_ATTN is the other
    # backend vLLM itself lists for this model and needs no nvcc.
    # It has to be the CLI flag: VLLM_ATTENTION_BACKEND no longer exists in
    # 0.27.1 -- setting it is silently ignored and FlashInfer is picked anyway.
    #
    # --kv-cache-dtype bfloat16 travels with it. Left to itself vLLM gives this
    # model an FP8 KV cache, which FlashInfer supports on sm_86 but Triton does
    # not ("native FP8 (fp8e4nv) requires SM89+"), so the engine refuses to
    # start. bf16 KV roughly halves the cache to ~23k tokens, still comfortably
    # above the 16,384 window.
    exec $GEMMA_VENV/bin/vllm serve RedHatAI/Qwen3.8-27B-INT4 \
        --host 0.0.0.0 --port 8011 \
        --allowed-local-media-path $MEDIA \
        --max-model-len 16384 --gpu-memory-utilization 0.93 \
        --enforce-eager --skip-mm-profiling \
        --attention-backend TRITON_ATTN \
        --kv-cache-dtype bfloat16 \
        --media-io-kwargs '{"video": {"num_frames": -1}}' \
        > "$LOGS/bench.server.qwen38.log" 2>&1
    ;;
*)
    echo "usage: serve.vllm.sh {cosmos|qwen3vl|llavavideo|gemma4|qwen38}" >&2
    exit 2
    ;;
esac
