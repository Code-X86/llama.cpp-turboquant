# llama.cpp + TurboQuant KV Cache

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp-green)](https://github.com/ggml-org/llama.cpp)

**Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)** with TurboQuant KV-cache vector quantization for AMD ROCm.

Compresses the KV cache to 3-4 bits per dimension using Walsh-Hadamard Transform + Lloyd-Max optimal quantization ([Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)). Reduces KV cache VRAM by 72-78%.

Prompt processing is at or above f16 — on head_dim 256 it is ~53% faster. Token generation at shallow context costs a few percent; at deep context it drops to ~0.63x f16, which is where quantized KV sits across methods and backends (see [Fused kernel](#fused-kernel)).

## Results

Benchmarks below: AMD RX 9070 XT (16 GB VRAM, gfx1201), ROCm 6.1, NixOS.

### Throughput — Qwen3-14B Q4_K_M (head_dim 128)

| KV Type | pp512 | pp8K | pp32K | tg128 | bpw | KV Savings |
|---------|-------|------|-------|-------|-----|------------|
| f16/f16 | 1894 | 1345 | 563 | 53.4 | 16.0 | — |
| q8_0/q8_0 | 1694 | 1340 | — | 52.1 | 8.0 | 50% |
| **turbo4/turbo4** | **1812** | **1321** | **550** | **49.3** | **4.5** | **72%** |
| **turbo3/turbo3** | **1836** | **1319** | **548** | **49.6** | **3.5** | **78%** |

### Throughput — Qwen3.5-9B Q8_0 (head_dim 256)

Prompt processing is *faster* than f16 here, not slower:

| type_k | type_v | pp512 | tg128 |
|--------|--------|-------|-------|
| f16 | f16 | 2565 ± 175 | 56.4 |
| **turbo4** | **turbo4** | **3929 ± 34** | **55.3** |
| **turbo3** | **turbo3** | **3933 ± 25** | **55.2** |

Any turbo operand lifts pp512 by ~53% while token generation stays flat. At 131072 context the KV cache drops from 4096 MiB (f16) to 896 MiB (turbo3) — 14.81 GB total VRAM down to 11.47 GB, which is the difference between fitting on this card and not.

### Quality (Perplexity)

Qwen3-14B Q4_K_M:

| KV Type | PPL | Delta vs f16 |
|---------|-----|-------------|
| f16 | 1.7031 | — |
| q8_0 | 1.7044 | +0.001 |
| **turbo4** | **1.7134** | **+0.010** |
| **turbo3** | **1.7544** | **+0.051** |

The error does not compound with context depth — the turbo3 gap is +0.0080 at 2048, +0.0065 at 8192 and +0.0066 at 16384. TurboQuant keeps the per-chunk L2 norm in fp16 and quantizes only direction, so the magnitude drift that [KVarN](https://arxiv.org/abs/2606.03458) attributes accumulation to does not exist here.

### Per-layer bit allocation

Layers do not benefit equally from an extra bit. Measured on Qwen3.5-9B (8 attention layers), at a fixed 4.0 bpw budget:

| where the extra bit goes | PPL |
|--------------------------|-----|
| nowhere — all turbo3 (3.5 bpw) | 1.5337 |
| the K side, or the V side | 1.5318 |
| **the right four layers** | **1.5292** |
| everywhere — all turbo4 (4.5 bpw) | 1.5284 |

Four of eight layers capture 85% of the full turbo4 gain at half the extra cost; six reach all of it at 4.25 bpw. K and V measure equally sensitive (0.0002 apart, i.e. noise) — the Hadamard transform spreads outlier channels across all 128 dimensions, flattening the K/V asymmetry other methods exploit.

Set it with `LLAMA_KV_TYPE_PER_LAYER` (see [Quick Start](#quick-start)). The ranking is model-specific; [docs/turboquant.md](docs/turboquant.md) has the full measurement and how to reproduce it.

## Quick Start

```bash
# Build (AMD ROCm)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run with TurboQuant KV cache
./build/bin/llama-cli -m model.gguf -fa 1 -ngl 99 \
  --cache-type-k turbo4 --cache-type-v turbo4

# Per-layer allocation: turbo4 on the layers that benefit, turbo3 elsewhere
LLAMA_KV_TYPE_PER_LAYER=turbo3,turbo3,turbo4,turbo4,turbo4,turbo3,turbo3,turbo4 \
  ./build/bin/llama-cli -m model.gguf -fa 1 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3

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

A row is processed in fixed 128-element chunks, so head_dim 256 is quantized as two independently normalized halves. Each half is a unit vector in R^128 — exactly the distribution the codebooks were fitted for — which is why the existing codebooks stay valid.

<a name="fused-kernel"></a>
### Fused kernel

K and V are decoded inside the flash-attention tile kernel rather than bulk-converted to f16 beforehand. The saving comes from Hadamard algebra: `<Hc, q> = <c, Hq>` lets Q be transformed once at kernel start, and `sum_j p_j H c_j = H (sum_j p_j c_j)` lets V accumulate untransformed and be transformed once at the end. Two transforms per launch replace two per KV token.

At depth 65536 this took token generation from 23.58 t/s (bulk conversion) to 29.59 t/s — 0.63x of f16. That matches what [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) reports for quantized KV at comparable depth; quantized KV is slower than f16 at decode across methods and backends.

A fused *vector* kernel also exists but is off by default (`GGML_CUDA_TURBO_FUSED_FA=1`): it handles one query head per block and so re-reads K and V `gqa_ratio` times, which cancels the compression advantage against the tile path.

## Requirements

- Flash Attention enabled (`-fa 1`)
- head_dim a multiple of 128 — 128 (Llama, Qwen, Mistral, Gemma, DeepSeek, Phi) and 256 (Qwen3.5) both tested
- AMD ROCm (gfx1201 tested) or CPU

## Limitations

- head_dim must be a multiple of 128
- CUDA compiles in CI but is untested on hardware — HIP/ROCm is the tested GPU path
- Mixed turbo + quantized V (e.g. turbo4/q8_0) not supported — use turbo/turbo or turbo/f16
- Token generation at deep context is ~0.63x f16 (see [Fused kernel](#fused-kernel))

## What Changed vs Upstream llama.cpp

See [docs/turboquant.md](docs/turboquant.md) for full details and measurements.

Key files:
- `ggml/src/ggml-cuda/set-rows.cu` — FWHT encode kernels (128 threads, shared-memory FWHT)
- `ggml/src/ggml-cuda/convert.cu` — FWHT decode kernels (bulk dequantize)
- `ggml/src/ggml-cuda/turbo.cuh` — shared decode primitives (unpack, FWHT, chunk decode)
- `ggml/src/ggml-cuda/fattn-tile.cuh` — K and V decoded inside flash attention
- `ggml/src/ggml-quants.c` — CPU reference implementation
- `src/llama-kv-cache.cpp` — head_dim guard, per-layer KV types
- `tests/test-turboquant.cpp` — 23 CPU reference tests across head_dim 128/256/384

CI (`.github/workflows/turboquant-ci.yml`) builds and tests on CPU and compile-checks CUDA with `-Werror` on every push and pull request. The upstream workflows this fork inherits are not a signal — several need runners a fork does not have, and some fail on upstream master too.

## License

MIT License — same as upstream llama.cpp. See [LICENSE](LICENSE).

TurboQuant algorithm: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026).

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org — the foundation this builds on
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) by Zandieh et al.
- [Reference implementation](https://github.com/0xSero/turboquant) by 0xSero
- [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — community research thread
