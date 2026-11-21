#!/bin/bash
# DeepSeek-V4-Flash-0731 on two GPUs: full-resident expert parallelism, 2M-token shared KV pool
# (2 concurrent requests, each up to the model's 1M context), FP8 compressed KV, 64 GiB host-RAM L2
# prefix cache. This is the configuration the README numbers were measured with.
#
# Required: DSV4_MODEL = checkpoint directory (deepseek-ai/DeepSeek-V4-Flash-0731 layout).
# GPU 0 = the large card (attention, router, shared expert, KV, head + part of the routed experts),
# GPU 1 = the second card (the remaining routed experts). See docs/USAGE.md.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${DSV4_MODEL:?set DSV4_MODEL to the DeepSeek-V4-Flash-0731 checkpoint directory}

# Start from a clean engine environment so no inherited switch leaks in.
for _v in $(compgen -e); do
  case $_v in FREETOKEN_*|PYTORCH_*|TORCH_*|TRITON_*) unset "$_v" ;; esac
done
unset _v

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH=$REPO/python
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Expert parallelism across the two cards.
export FREETOKEN_DSV4_EP=1
# Routed experts per layer placed on GPU 1 (124 leaves the RTX PRO 5000 72 GB room for the L2 page tables
# and the indexer split workspace).
export FREETOKEN_DSV4_EP_N1=${FREETOKEN_DSV4_EP_N1:-124}
# Indexer query split: share of each deep prefill indexer call scored on GPU 1 (0 = off).
export FREETOKEN_DSV4_IDX_SPLIT=${FREETOKEN_DSV4_IDX_SPLIT:-0.4}

# Kernels.
export FREETOKEN_DSFP4_GEMV=v2
export FREETOKEN_DSV4_WOA_FP8=1
export FREETOKEN_DSV4_HEAD_FP8=1
export FREETOKEN_DSV4_MXFP8_CUBLAS=1
export FREETOKEN_DSV4_IDX_KERNEL=fp4
export FREETOKEN_DSV4_IDX_FP4T=1
export FREETOKEN_DSV4_IDX_DEC_CUDA=1
export FREETOKEN_DSV4_TOPK_CUDA=1
export FREETOKEN_DSV4_TOPK_CUDA_DEC=1
export FREETOKEN_DSV4_ATTN_BF16=1
export FREETOKEN_DSV4_ATTN_CUDA=1
export FREETOKEN_DSV4_ATTN_CUDA_DEC=0
export FREETOKEN_DSV4_SHARED_CMP=1

# KV layout: ratio-4 compressed KV stored FP8, indexer keys FP4, all 2M tokens in VRAM.
export FREETOKEN_DSV4_CMP_FP8=1
export FREETOKEN_DSV4_IDX_FP4POOL=1
export FREETOKEN_DSV4_CMP_BF16_TOKENS=0
export FREETOKEN_DSV4_CMP_GPU_TOKENS=2097152

# Prefill: 8192-token chunks run the in-chunk two-micro-batch EP pipeline (needs >= 142 window pages).
export FREETOKEN_PREFILL_CHUNK_CAP=8192

# Host-RAM L2 prefix cache (GiB of pinned memory).
export FREETOKEN_DSV4_L2_GB=${FREETOKEN_DSV4_L2_GB:-64}
# Share of the L2 for window pages (one per resume point, ~17 MiB each).
export FREETOKEN_DSV4_L2_WIN_FRAC=${FREETOKEN_DSV4_L2_WIN_FRAC:-0.3}

# Request defaults; deploy/runtime.json overrides them at runtime without a restart.
export FREETOKEN_DEFAULT_MAX_OUTPUT_TOKENS=393216
export FREETOKEN_RUNTIME_CONF=${FREETOKEN_RUNTIME_CONF:-$REPO/deploy/runtime.json}

exec ${FT:-ft} serve \
  --model "$MODEL" \
  --served-model-name "${DSV4_NAME:-dsv4-flash}" \
  --host "${DSV4_HOST:-127.0.0.1}" --port "${DSV4_PORT:-1920}" \
  --moe-backend offload \
  --memory-ratio 0.95 \
  --kv-reserve-tokens 2097152 \
  --max-seq-len-override 1048576 \
  --max-running-requests ${DSV4_MAX_RUNNING:-4} \
  --num-tokens 2097152 \
  --swa-num-tokens 142
