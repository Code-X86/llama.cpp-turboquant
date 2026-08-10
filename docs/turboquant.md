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

### Throughput — Qwen3.5-9B Q8_0 (head_dim 256)

RX 9070 XT, `-fa 1 -ngl 99`, 2 runs each. Note the reversed picture compared to
head_dim 128: turbo is substantially *faster* at prompt processing here.

| type_k | type_v | pp512 | tg128 |
|--------|--------|-------|-------|
| f16 | f16 | 2565 ± 175 | 56.4 |
| f16 | turbo4 | 4084 ± 118 | 55.8 |
| turbo4 | turbo4 | 3929 ± 34 | 55.3 |
| turbo4 | f16 | 3981 ± 28 | 55.7 |
| turbo3 | turbo3 | 3933 ± 25 | 55.2 |
| turbo3 | f16 | 3966 ± 17 | 55.9 |

Any turbo operand lifts pp512 by ~53% over f16/f16 while token generation stays
flat. The likely cause is a different flash-attention kernel being selected once
an operand is a turbo type — unverified, worth a look with `GGML_CUDA_DEBUG`.

KV cache at 131072 context, 8 attention layers (Qwen3.5 is hybrid — the other
24 layers are linear attention with a constant-size recurrent state):

| KV Type | KV cache | Total VRAM | Savings |
|---------|----------|-----------|---------|
| f16 | 4096 MiB | 14.81 GB | — |
| turbo4 | 1152 MiB | 11.75 GB | 72% |
| turbo3 | 896 MiB | 11.47 GB | 78% |

At 131072 context f16 needs 14.81 GB of the card's 16.3 GB, leaving almost no
headroom for the vision encoder. turbo3 brings that down to 11.47 GB.

### Token generation degrades with context depth

The pre-dequantize strategy converts the whole KV cache to f16 once per FA call.
That cost is O(n_kv) per generated token, so it grows with context depth while the
attention work per token stays flat. Measured on Qwen3.5-9B Q8_0, RX 9070 XT:

| type_k | type_v | tg128 @ d0 | tg128 @ d16384 | tg128 @ d65536 |
|--------|--------|-----------|---------------|---------------|
| f16 | f16 | 56.8 | 54.1 | **46.9** |
| f16 | turbo4 | 55.9 | 48.0 | 32.3 |
| f16 | turbo3 | 56.1 | 47.3 | 31.5 |
| turbo4 | f16 | 55.7 | 48.0 | 32.5 |
| turbo3 | f16 | 56.4 | 48.2 | 31.9 |
| turbo3 | turbo4 | 55.9 | 43.1 | **24.5** |

One quantized cache costs ~32 t/s at depth 65536, two cost ~24.5, f16 gets 46.9.
The penalty tracks the *number* of converted caches, not which one — exactly what
a per-cache bulk conversion predicts.

Prompt processing is unaffected (turbo is in fact faster there, 1012 vs 730 t/s at
d65536) because the single conversion amortizes over the whole batch.

### Fused kernel: implemented, correct, currently slower

A fused vector kernel that decodes turbo blocks inside flash attention exists
(`fattn-vec-turbo.cuh`). It removes the bulk conversion entirely by working in
the Hadamard domain — Q is transformed once at kernel start, V is accumulated
untransformed and transformed once at the end, so two transforms per launch
replace two per KV token.

It is correct (test-backend-ops FLASH_ATTN_EXT: turbo3 640/640, turbo4 640/640)
but **slower than the path it replaces**, measured A/B in the same build:

| depth | fused (KQ=32) | fused (KQ=8) | bulk conversion | f16 |
|-------|--------------|-------------|-----------------|-----|
| 0 | 52.5 | 53.2 | 55.0 | 56.8 |
| 16384 | 35.7 | 39.2 | 42.5 | 54.1 |
| 65536 | 16.6 | **20.4** | **23.6** | **46.9** |

Narrowing the KQ reduction from 32 lanes per token to 8 (one warp reduction is
then 3 stages instead of 5, and 4 tokens are dotted at once) bought 23% at depth
65536. It does not close the gap, because the remaining cause is structural:

**GQA amortization.** For `Q->ne[1] == 1` with `gqa_opt_applies`, f16 does not
use the vector kernel at all — `fattn.cu` routes it to the tile kernel, which
processes all `gqa_ratio` query heads per KV read. The fused vector kernel
handles one query head per block and therefore reads K and V `gqa_ratio` times
as often. Per element and query head, with Qwen3.5's gqa_ratio = 4:

    tile + f16:     2 bytes / 4 heads = 0.50 bytes
    fused + turbo3: 0.4375 bytes x 1  = 0.4375 bytes

The 4.57x compression collapses to a 12% bandwidth edge, which the unpacking and
codebook lookups more than consume. No amount of tuning inside the vector kernel
recovers this; a fused kernel has to support GQA to beat the tile path.

Therefore **disabled by default**. Enable with `GGML_CUDA_TURBO_FUSED_FA=1`.

### Fused tile V path: also no gain

The tile kernel is what f16 actually uses for decode with GQA, and its V tile
holds complete contiguous head_dim rows, so decoding there avoids the vector
kernel's GQA penalty entirely. That path exists too (`type_V` threaded through
`flash_attn_tile`, `flash_attn_tile_load_tile_turbo`, `need_f16_V = false`) and
is correct — 640/640 on both turbo types. It does not help either, at depth 65536:

| config | bulk | fused V |
|--------|------|---------|
| f16 / turbo3 | 31.46 | 31.08 |
| turbo3 / turbo3 | 23.58 | 23.54 |

Traffic does drop from 2 to 0.4375 bytes per element. The kernel is not waiting
on that. The original tile load is a plain copy, fully pipelined across all
threads; decoding instead walks chunk by chunk through an FWHT whose five shuffle
stages form a dependency chain. It is latency-bound, not bandwidth-bound.

**Conclusion after two independent attempts:** TurboQuant decoding is too
expensive to perform inside attention. The bulk conversion amortizes it over the
whole cache once per FA call, which neither a vector nor a tile kernel can match
by decoding on demand. The fused tile V path is kept because it removes the f16
staging buffer for V (~128 MB at 65536 context) at no throughput cost.

If this is revisited, the thing to attack is the decode cost itself, not where it
runs: hold the codebook lane-resident and read it via `__shfl_sync` instead of
memory, and check whether the XOR-16 shuffle stage lowers to `ds_bpermute` (it
crosses the DPP16 row boundary on RDNA 4). Both target the dependency chain that
makes decoding slow in the first place.

### Perplexity (lower is better)

Measured on ~960KB C++ source code corpus, context=2048.

| KV Type | PPL | Delta vs f16 |
|---------|-----|-------------|
| f16 | 1.7031 | — |
| q8_0 | 1.7044 | +0.001 |
| turbo4 | 1.7134 | +0.010 |
| turbo3 | 1.7544 | +0.051 |

Qwen3.5-9B Q8_0 (head_dim 256), 600KB C++ corpus, context=2048:

| KV Type | PPL | Delta vs f16 |
|---------|-----|-------------|
| f16 | 1.5258 ± 0.0071 | — |
| turbo4 | 1.5281 ± 0.0071 | +0.002 |
| turbo3 | 1.5343 ± 0.0072 | +0.009 |

Both deltas are markedly smaller than at head_dim 128 (+0.010 / +0.051). Splitting
a 256-dim head into two independently normalized halves stores two norms instead
of one, which tracks variance differences between the halves more closely — at no
extra cost, since the norm is replicated per block either way.

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
