# Heterogeneous Inference: DeepSeek-V4-Flash

DeepSeek-V4-Flash-0731 served from two different NVIDIA Blackwell workstation cards, an RTX PRO 6000 (96 GB) and an RTX PRO 5000 (72 GB), connected over PCIe without NVLink. All routed experts stay resident in VRAM, and two requests can each use the full 1M-token context at the same time. The engine is a modified [FreeToken](https://github.com/FlashML-org/FreeToken).

## What is in it

- **Expert parallelism across unequal cards.** Routed experts are split per layer between the two GPUs. Decode is a graph-captured fork/join with peer-to-peer stores. Dense projections that would leave the second card idle are split by measured bandwidth. The expert-parallel split reproduces the single-GPU kernels bit for bit.
- **2M-token shared KV pool, entirely in VRAM.** Compressed KV is packed to FP8 and indexer keys are stored as FP4, so 2 × 1M tokens fit next to the weights.
- **Long-context prefill kernels.** These include a CUDA FP4 indexer, radix top-k, 64-head sparse MLA prefill, cuBLASLt MXFP8 GEMMs, and a two-micro-batch pipeline that overlaps one card's attention with the other card's experts.
- **Host-RAM L2 prefix cache.** Sessions evicted from VRAM are kept in 64 GiB of pinned memory and restored on the next request.
- **Hot-reloadable request defaults.** Default output budget, thinking mode, sampling and reasoning effort are read from a JSON file on every request, so changing them needs no restart.
- **Request filter.** A proxy in front of the engine accepts only the inference endpoints.

## Results

| Workload | Time to first token | Decode |
|---|---:|---:|
| 1K prompt, 1 stream | 0.23 s | 102.1 tok/s |
| 8K prompt, 1 stream | 1.10 s | 101.2 tok/s |
| 1K prompt, 2 streams | 0.25 s | 137.3 tok/s total |
| 128K prompt, cold | 18.5 s | 97.8 tok/s |
| 1M prompt (1,010,967 tokens), cold | 258.7 s | 91.7 tok/s |
| 2 × 1M, decoding together | — | 121.6 tok/s total |

These numbers were measured on 2026-11-19 with `deploy/serve.sh` on an RTX PRO 6000 and an RTX PRO 5000 Blackwell (PRO 6000 power limit 400 W). The checkpoint was a DeepSeek-V4-Flash-0731 derivative with an identical architecture and config. The run took place before the host L2 cache was enabled.

## Documentation

[docs/USAGE.md](docs/USAGE.md)

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
