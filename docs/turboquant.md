# TurboQuant KV Cache Quantization (Experimental)

TurboQuant compresses the KV cache to 3-4 bits per dimension using
Walsh-Hadamard Transform + optimal vector quantization.

Based on: Zandieh, Daliri, Hadian, Mirrokni — "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate", ICLR 2026.
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## Usage

```bash
# TurboQuant 4-bit (4.5 bpw) — recommended default
llama-cli -m model.gguf -fa 1 -ngl 99 --cache-type-k turbo4 --cache-type-v turbo4

# TurboQuant 3-bit (3.5 bpw) — maximum compression
llama-cli -m model.gguf -fa 1 -ngl 99 --cache-type-k turbo3 --cache-type-v turbo3

# Mixed: turbo K with f16 V (or vice versa)
llama-cli -m model.gguf -fa 1 -ngl 99 --cache-type-k turbo4 --cache-type-v f16
```

Flash attention (`-fa 1`) is required.

## Performance

All benchmarks: Qwen3-14B Q4_K_M, AMD RX 9070 XT (16 GB VRAM, gfx1201), ROCm 6.1, NixOS.

### Throughput (tokens/second)

| KV Type | pp512 | pp2K | pp8K | pp16K | pp32K | pp40K | tg128 | bpw |
|---------|-------|------|------|-------|-------|-------|-------|-----|
| f16/f16 | 1894 | 1847 | 1345 | 943 | 563 | 471 | 53.4 | 16.0 |
| q8_0/q8_0 | 1694 | 1841 | 1340 | — | — | — | 52.1 | 8.0 |
| **turbo4/turbo4** | **1812** | **1816** | **1321** | **926** | **550** | **460** | **49.3** | **4.5** |
| **turbo3/turbo3** | **1836** | **1816** | **1319** | **924** | **548** | — | **49.6** | **3.5** |

### Perplexity (lower is better)

Measured on ~960KB C++ source code corpus, context=2048.

| KV Type | PPL | Delta vs f16 |
|---------|-----|-------------|
| f16 | 1.7031 | — |
| q8_0 | 1.7044 | +0.001 |
| turbo4 | 1.7134 | +0.010 |
| turbo3 | 1.7544 | +0.051 |

### KV Cache Memory

| KV Type | Bytes per element | Savings vs f16 |
|---------|------------------|----------------|
| f16 | 2.000 | — |
| q8_0 | 1.000 | 50% |
| turbo4 | 0.5625 (18/32) | 72% |
| turbo3 | 0.4375 (14/32) | 78% |

## How it works

```
Encode: x → ||x|| → x/||x|| → FWHT(x/||x||) → nearest_centroid() → pack_bits → (norm_fp16, packed_indices)
Decode: unpack → centroid_lookup → inverse_FWHT → × norm → x̃
```

The Fast Walsh-Hadamard Transform (FWHT) rotates the input vector into a domain
where the Lloyd-Max codebook is optimal. The codebook centroids are precomputed
for the Beta((d-1)/2, (d-1)/2) distribution that arises after FWHT of unit vectors
in d=128 dimensions.

The transform is always applied in fixed 128-element chunks, walking the row.
A head larger than 128 is therefore split into independently normalized chunks:
a 256-dim head (Qwen3.5) becomes two halves, each with its own L2 norm and its
own FWHT. Because every normalized half is again a unit vector in R^128, the
same codebooks stay optimal — no per-head-size codebook is needed. The stored
norm is already replicated per 32-element block, so the extra half-norm costs
no additional bytes and the bit rate is unchanged.

Current implementation uses a pre-dequantize strategy: turbo KV data is bulk-converted
to f16 before standard Flash Attention runs. This adds minimal overhead (~3% pp, ~8% tg)
while avoiding the complexity of a fused FA kernel.

## Requirements

- Flash Attention enabled (`-fa 1`)
- head_dim must be a multiple of 128 — 128 covers Llama, Qwen3, Mistral and
  Gemma; 256 covers Qwen3.5
- AMD ROCm (gfx1201 tested) or CPU

## Limitations

- head_dim must be a multiple of 128 (FWHT chunk size). Models with head_dim
  80, 96 or 112 are not supported.
- Mixed turbo/quantized combinations (e.g. turbo4/q8_0) are not supported — use turbo/turbo or turbo/f16
- No CUDA support yet (HIP/ROCm only for GPU path)
- Pre-dequantize strategy means full KV cache is converted to f16 per FA call (a fused kernel would eliminate this)

## GGML Types

| Type | GGML ID | Block Size | Block Bytes | bpw |
|------|---------|-----------|-------------|-----|
| GGML_TYPE_TURBO3_0 | 41 | 32 | 14 (2 norm + 12 packed) | 3.5 |
| GGML_TYPE_TURBO4_0 | 42 | 32 | 18 (2 norm + 16 packed) | 4.5 |

## Build

```bash
# AMD ROCm (RX 9070 XT)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
./build/bin/test-turboquant

# Benchmark
./build/bin/llama-bench -m model.gguf -fa 1 -ngl 99 -ctk turbo4 -ctv turbo4 -p 512 -n 128
```
