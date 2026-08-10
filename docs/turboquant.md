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

### Fused tile path: K and V decoded inside attention

The tile kernel is what f16 uses for decode with GQA, so decoding there avoids
the vector kernel's GQA penalty. Both sides are now fused (`type_K`/`type_V`
threaded through `flash_attn_tile` / `_iter` / `_iter_KQ`, `need_f16_K/V = false`).

The decisive detail is *where* the Hadamard transform happens. A first version
decoded each chunk as it was loaded — one inverse FWHT per KV token, the same
O(n_kv) cost as the bulk conversion, just relocated. It measured exactly as fast
as what it replaced. Working in the Hadamard domain instead reduces that to O(1):

  V:  sum_j p_j H c_j = H (sum_j p_j c_j)   accumulate raw, transform once at the end
  K:  <H c, q> = <c, H q>                   transform Q once at kernel start

tg128 at depth 65536, Qwen3.5-9B Q8_0 on gfx1201:

| stage | t/s | of f16 |
|-------|-----|--------|
| bulk conversion (baseline) | 23.58 | 0.50x |
| V fused | 26.62 | 0.57x |
| K and V fused | **29.59** | **0.63x** |
| f16 reference | 46.90 | 1.00x |

0.63x matches what llama.cpp discussion #20969 reports for quantized KV at 110K
context. Quantized KV is slower than f16 at decode across methods and backends —
that is a property of the approach, not of this implementation. What was
recoverable has been recovered.

Two bugs worth recording, both found by the test matrix rather than by reading:

- The K tile loader assumed a slice narrower than one 128-element chunk. True for
  `nbatch_K = 64`, false for 128 and 256, which the config table also selects.
  The chunk (and its norm) is now derived per element.
- Staging all of Q for the transform needed `ncols*DKQ` floats in `KV_tmp`, which
  is sized by `nbatch_fa*nbatch_K` — and those shrink as `ncols` grows. At
  `ncols=64` the buffer was short by a factor of 1.8. Transforming one chunk at a
  time needs 128 floats regardless of `ncols`. This is why only large batches
  failed while `nb=1` stayed green.

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

#### The error does not accumulate with context depth

KVarN (arXiv 2606.03458, llama.cpp #24139) argues that KV-cache quantization
error compounds during autoregressive decoding, so a short-context perplexity
number can understate the real cost. Measured against that concern:

| context | f16 | turbo4 | delta | turbo3 | delta |
|---------|-----|--------|-------|--------|-------|
| 2048 | 1.5258 | 1.5276 | +0.0018 | 1.5338 | +0.0080 |
| 8192 | 1.4107 | 1.4124 | +0.0017 | 1.4172 | +0.0065 |
| 16384 | 1.3431 | not measured | — | 1.3497 | +0.0066 |

The gap does not widen. For turbo3 it drops from +0.0080 at 2048 to +0.0065 at
8192 and then holds at +0.0066 through 16384 — 0.52%, 0.46%, 0.49% in relative
terms. Three depths across an 8x range, flat after the first step.

(turbo4 at 16384 is missing; its delta is ~0.0017 at both measured depths, small
enough that the trend question hardly applies.)

That is consistent with how TurboQuant stores data: the per-chunk L2 norm is kept
in fp16 and never quantized, so only the *direction* of the normalized vector is
approximated. KVarN attributes the accumulating error specifically to incorrect
per-token magnitude scaling — the mechanism does not exist here. The per-chunk
normalization that makes TurboQuant expensive at runtime is the same property
that makes it robust over long contexts.

### Per-layer bit allocation

Layers do not benefit equally from the extra bit. RateQuant (arXiv 2605.06675)
derives an uneven allocation from a fitted distortion curve; the numbers below
measure it directly instead, on Qwen3.5-9B (8 attention layers, 24 linear-attention
layers with no KV cache).

Each row upgrades exactly one layer to turbo4 and leaves the other seven on
turbo3, against a 1.5337 all-turbo3 baseline:

| layer | PPL | delta |
|-------|-----|-------|
| 0 | 1.5341 | +0.0004 |
| 1 | 1.5338 | +0.0001 |
| 2 | 1.5324 | -0.0013 |
| 3 | 1.5325 | -0.0012 |
| 4 | 1.5321 | -0.0016 |
| 5 | 1.5330 | -0.0007 |
| 6 | 1.5329 | -0.0008 |
| 7 | 1.5326 | -0.0011 |

Layer 0 gets *worse* with more bits, which is impossible and puts the noise floor
at roughly +/-0.0005. Every individual number is therefore within one to three
times the noise — far too weak to allocate bits from on its own.

The ranking they form, however, is not. Two runs at an identical 4.0 bpw, differing
only in which four layers are upgraded:

| allocation | bpw | PPL |
|------------|-----|-----|
| all turbo3 | 3.500 | 1.5337 |
| worst four (0,1,5,6) | 4.000 | 1.5322 |
| best four (2,3,4,7) | 4.000 | 1.5292 |
| all turbo4 | 4.500 | 1.5284 |

0.0030 separates two configurations that cost exactly the same — six times the
noise floor. The ranking carries real information even though the measurements it
was built from individually do not; the errors partly cancel when sorting.

Walking the ranking (best first: 4, 2, 3, 7, 6, 5, 0, 1) gives the quality/size
curve:

| upgraded layers | bpw | PPL | share of the turbo4 gain |
|-----------------|-----|-----|--------------------------|
| 0 | 3.500 | 1.5337 | 0% |
| 2 | 3.750 | 1.5311 | 49% |
| 3 | 3.875 | 1.5303 | 64% |
| 4 | 4.000 | 1.5292 | 85% |
| 6 | 4.250 | 1.5279 | 109% |
| 8 | 4.500 | 1.5284 | 100% |

Four of eight layers capture 85% of what all eight provide, at half the extra
cost. Six reach full turbo4 quality at 4.25 bpw instead of 4.5.

That six beat eight is 0.0005, exactly the noise floor, so the ordering between
those two is not itself meaningful — but the two remaining layers are 0 and 1,
the only two whose single-layer measurement came out positive. Two independent
measurements agree that they gain nothing, which is worth more than either alone.

Two useful operating points, then: 4.0 bpw for 85% of the gain, or 4.25 bpw for
all of it — against 4.5 bpw for uniform turbo4.

Set with `LLAMA_KV_TYPE_PER_LAYER`, a comma-separated ggml type name per KV layer
(K and V get the same type). Fewer entries than layers: the last one applies to
the remainder, with a warning.

```bash
# best four layers of Qwen3.5-9B at turbo4, rest at turbo3 (4.0 bpw average)
LLAMA_KV_TYPE_PER_LAYER=turbo3,turbo3,turbo4,turbo4,turbo4,turbo3,turbo3,turbo4 \
  llama-cli -m model.gguf -fa on --cache-type-k turbo3 --cache-type-v turbo3
```

The ranking is model-specific — it was measured on this model and should not be
assumed to transfer. Reproduce it for another model by upgrading one layer at a
time, as above.

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
