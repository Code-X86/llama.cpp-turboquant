# llama.cpp + TurboQuant KV Cache

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp-green)](https://github.com/ggml-org/llama.cpp)

**Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** with TurboQuant KV-cache vector quantization for AMD ROCm.

Compresses the KV cache to 3-4 bits per dimension using Walsh-Hadamard Transform + Lloyd-Max optimal quantization ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)). Reduces KV cache VRAM by 72-78% with less than 10% performance overhead.

## Results

All benchmarks: Qwen3-14B Q4_K_M, AMD RX 9070 XT (16 GB VRAM, gfx1201), ROCm 6.1, NixOS.

### Throughput

| KV Type | pp512 | pp8K | pp32K | tg128 | bpw | KV Savings |
|---------|-------|------|-------|-------|-----|------------|
| f16/f16 | 1894 | 1345 | 563 | 53.4 | 16.0 | — |
| q8_0/q8_0 | 1694 | 1340 | — | 52.1 | 8.0 | 50% |
| **turbo4/turbo4** | **1812** | **1321** | **550** | **49.3** | **4.5** | **72%** |
| **turbo3/turbo3** | **1836** | **1319** | **548** | **49.6** | **3.5** | **78%** |

### Quality (Perplexity)

| KV Type | PPL | Delta vs f16 |
|---------|-----|-------------|
| f16 | 1.7031 | — |
| q8_0 | 1.7044 | +0.001 |
| **turbo4** | **1.7134** | **+0.010** |
| **turbo3** | **1.7544** | **+0.051** |

## Quick Start

```bash
# Build (AMD ROCm)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run with TurboQuant KV cache
./build/bin/llama-cli -m model.gguf -fa 1 -ngl 99 \
  --cache-type-k turbo4 --cache-type-v turbo4

# Benchmark
./build/bin/llama-bench -m model.gguf -fa 1 -ngl 99 \
  -ctk turbo4 -ctv turbo4 -p 512 -n 128

# Run tests
./build/bin/test-turboquant
```

Flash attention (`-fa 1`) is required.

## Available Types

| Type | Bits | Block | Bytes | Use Case |
|------|------|-------|-------|----------|
| `turbo4` | 4-bit | 32 | 18 | Recommended default — near-lossless |
| `turbo3` | 3-bit | 32 | 14 | Maximum compression — slightly higher PPL |

Mixed configurations work: `--cache-type-k turbo4 --cache-type-v f16` (or vice versa).

## How It Works

```
Encode: x → L2norm → x/‖x‖ → FWHT(128) → scalar quantize → bitpack → (norm_fp16, indices)
Decode: unpack → codebook lookup → inverse FWHT → ×norm → x̃
```

The Fast Walsh-Hadamard Transform rotates input vectors into a domain where Lloyd-Max scalar quantization is optimal. The codebook centroids are precomputed for the Beta((d-1)/2, (d-1)/2) distribution arising after FWHT of unit vectors in d=128 dimensions.

Current implementation uses pre-dequantize strategy: turbo KV data is bulk-converted to f16 before standard Flash Attention. This adds minimal overhead while avoiding the complexity of a fused FA kernel.

## Requirements

- Flash Attention enabled (`-fa 1`)
- head_dim = 128 (Llama, Qwen, Mistral, Gemma, DeepSeek, Phi, and most current models)
- AMD ROCm (gfx1201 tested) or CPU

## Limitations

- head_dim must be exactly 128
- No CUDA support yet (HIP/ROCm only for GPU path)
- Mixed turbo + quantized V (e.g. turbo4/q8_0) not supported — use turbo/turbo or turbo/f16

## What Changed vs Upstream llama.cpp

22 files changed, ~1550 lines added. See [docs/turboquant.md](docs/turboquant.md) for full details.

Key files:
- `ggml/src/ggml-cuda/set-rows.cu` — FWHT encode kernels (128 threads, shared-memory FWHT)
- `ggml/src/ggml-cuda/convert.cu` — FWHT decode kernels (bulk dequantize)
- `ggml/src/ggml-cuda/fattn.cu` — Pre-dequantize integration in Flash Attention
- `ggml/src/ggml-quants.c` — CPU reference implementation
- `src/llama-kv-cache.cpp` — head_dim=128 validation guard
- `tests/test-turboquant.cpp` — 7 CPU reference tests

## License

MIT License — same as upstream llama.cpp. See [LICENSE](LICENSE).

TurboQuant algorithm: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026).

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org — the foundation this builds on
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) by Zandieh et al.
- [Reference implementation](https://github.com/0xSero/turboquant) by 0xSero
- [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — community research thread
