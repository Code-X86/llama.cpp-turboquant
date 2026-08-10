#pragma once

#include "common.cuh"
#include "dequantize.cuh"

// ============================================================
// TurboQuant shared building blocks
// ============================================================
//
// TurboQuant stores a KV row as fixed 128-element chunks. Each chunk is
// L2-normalized, rotated by a Walsh-Hadamard transform and scalar-quantized
// against a Lloyd-Max codebook:
//
//     stored:      c = quantize(H x / (||x|| sqrt(128)))    (codebook indices + ||x||)
//     recovered:   x = ||x|| / sqrt(128) * H c
//
// H is the unnormalized +-1 Hadamard matrix, which is symmetric and orthogonal.
// A single element therefore cannot be decoded on its own — it needs the whole
// 128-element chunk. Everything below operates on complete chunks.

#define TURBO_HEAD_DIM_GPU         128
#define TURBO_BLOCKS_PER_CHUNK_GPU (TURBO_HEAD_DIM_GPU / 32)  // 4

// 1/sqrt(128) — matches turbo_fwht_f32 in ggml-quants.c
#define TURBO_FWHT_SCALE_GPU 0.08838834764831844f

// Unpack one 3-bit codebook index. Mirrors turbo_unpack3 in ggml-quants.c.
static __device__ __forceinline__ uint8_t turbo3_unpack_index(const block_turbo3_0 * x, int elem_in_block) {
    const uint8_t * qs = x->qs;

    const int bit_off  = elem_in_block * 3;
    const int byte_idx = bit_off / 8;
    const int shift    = bit_off % 8;

    uint16_t raw = (uint16_t)qs[byte_idx] >> shift;
    if (shift > 5 && byte_idx + 1 < 12) {
        raw |= (uint16_t)qs[byte_idx + 1] << (8 - shift);
    }
    return (uint8_t)(raw & 0x07);
}

static __device__ __forceinline__ uint8_t turbo4_unpack_index(const block_turbo4_0 * x, int elem_in_block) {
    const uint8_t packed = x->qs[elem_in_block / 2];
    return (elem_in_block & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
}

// ------------------------------------------------------------
// FWHT, shared-memory variant: 128 threads, one element each
// ------------------------------------------------------------
//
// Stage order and pairing (i, i+h) are deliberately identical to
// turbo_fwht_f32 in ggml-quants.c so both produce the same rounding.
static __device__ __forceinline__ void turbo_fwht_smem_128(float * smem, int tid) {
    for (int h = 1; h < TURBO_HEAD_DIM_GPU; h *= 2) {
        if (tid < TURBO_HEAD_DIM_GPU/2) {
            const int group = tid / h;
            const int pos   = tid % h;
            const int i     = group * h * 2 + pos;
            const float a = smem[i];
            const float b = smem[i + h];
            smem[i]     = a + b;
            smem[i + h] = a - b;
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------
// FWHT, warp variant: 32 lanes x 4 consecutive elements
// ------------------------------------------------------------
//
// Lane l holds elements [4l, 4l+4) of the chunk, so the element index splits as
//     i = 4*lane + r,   r in [0,4)
// with bits 0..1 selecting the register and bits 2..6 selecting the lane. The
// first two butterfly stages (h = 1, 2) are therefore register-local, the
// remaining five (h = 4..64) are lane exchanges with XOR mask h/4.
//
// The arithmetic per butterfly is the same as in the shared-memory variant: the
// lower partner computes a+b, the upper computes a-b, in that order. Results are
// bit-identical to turbo_fwht_smem_128, which is what makes the fused kernel
// verifiable against the bulk dequantize path.
//
// Requires a full 32-lane warp with all lanes active.
static __device__ __forceinline__ void turbo_fwht_warp_128(float v[4], int lane) {
    // h = 1: pairs (0,1) and (2,3) inside the register file
    {
        const float a0 = v[0], b0 = v[1];
        const float a1 = v[2], b1 = v[3];
        v[0] = a0 + b0; v[1] = a0 - b0;
        v[2] = a1 + b1; v[3] = a1 - b1;
    }
    // h = 2: pairs (0,2) and (1,3)
    {
        const float a0 = v[0], b0 = v[2];
        const float a1 = v[1], b1 = v[3];
        v[0] = a0 + b0; v[2] = a0 - b0;
        v[1] = a1 + b1; v[3] = a1 - b1;
    }
    // h = 4, 8, 16, 32, 64: cross-lane, XOR mask h/4
#pragma unroll
    for (int mask = 1; mask < 32; mask *= 2) {
        const bool upper = (lane & mask) != 0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float p = __shfl_xor_sync(0xffffffff, v[r], mask, 32);
            v[r] = upper ? (p - v[r]) : (v[r] + p);
        }
    }
}

// ------------------------------------------------------------
// Chunk decode into shared memory (128 threads per chunk)
// ------------------------------------------------------------

template <typename block_t>
static __device__ __forceinline__ void turbo_decode_chunk_smem(
        const block_t * blocks, int64_t ib_chunk, float * smem, int tid);

template <>
__device__ __forceinline__ void turbo_decode_chunk_smem<block_turbo3_0>(
        const block_turbo3_0 * blocks, int64_t ib_chunk, float * smem, int tid) {
    const int local_block   = tid / TURBO3_BLOCK_SIZE;
    const int elem_in_block = tid % TURBO3_BLOCK_SIZE;

    smem[tid] = dc_codebook_3bit[turbo3_unpack_index(blocks + ib_chunk + local_block, elem_in_block)];
    __syncthreads();

    turbo_fwht_smem_128(smem, tid);

    // The norm is replicated in every block of the chunk; read it from the first.
    const float norm = __half2float(blocks[ib_chunk].d);
    smem[tid] *= TURBO_FWHT_SCALE_GPU * norm;
    __syncthreads();
}

template <>
__device__ __forceinline__ void turbo_decode_chunk_smem<block_turbo4_0>(
        const block_turbo4_0 * blocks, int64_t ib_chunk, float * smem, int tid) {
    const int local_block   = tid / TURBO4_BLOCK_SIZE;
    const int elem_in_block = tid % TURBO4_BLOCK_SIZE;

    smem[tid] = dc_codebook_4bit[turbo4_unpack_index(blocks + ib_chunk + local_block, elem_in_block)];
    __syncthreads();

    turbo_fwht_smem_128(smem, tid);

    const float norm = __half2float(blocks[ib_chunk].d);
    smem[tid] *= TURBO_FWHT_SCALE_GPU * norm;
    __syncthreads();
}

// ------------------------------------------------------------
// Chunk decode with an arbitrary thread count
// ------------------------------------------------------------
//
// The tile kernel runs with fewer threads than the 128 a chunk has elements
// (64 for head_dim 256 on RDNA), so the fixed one-thread-per-element form above
// does not fit. This variant strides over the chunk instead. The butterfly
// stages keep the same order and operand order, so results stay bit-identical
// to turbo_fwht_smem_128.
//
// All nthreads threads of the block must call this; it synchronizes internally.

template <int nthreads>
static __device__ __forceinline__ void turbo_fwht_smem_128_n(float * smem, int tid) {
    for (int h = 1; h < TURBO_HEAD_DIM_GPU; h *= 2) {
#pragma unroll
        for (int b0 = 0; b0 < TURBO_HEAD_DIM_GPU/2; b0 += nthreads) {
            const int b = b0 + tid;
            if (nthreads >= TURBO_HEAD_DIM_GPU/2 && b >= TURBO_HEAD_DIM_GPU/2) {
                continue;
            }
            const int group = b / h;
            const int pos   = b % h;
            const int i     = group * h * 2 + pos;
            const float a = smem[i];
            const float c = smem[i + h];
            smem[i]     = a + c;
            smem[i + h] = a - c;
        }
        __syncthreads();
    }
}

template <typename block_t, int nthreads>
static __device__ __forceinline__ void turbo_decode_chunk_smem_n(
        const block_t * blocks, int64_t ib_chunk, float * smem, int tid) {
    constexpr int bs = (sizeof(block_t) == sizeof(block_turbo3_0)) ? TURBO3_BLOCK_SIZE : TURBO4_BLOCK_SIZE;

#pragma unroll
    for (int e0 = 0; e0 < TURBO_HEAD_DIM_GPU; e0 += nthreads) {
        const int e = e0 + tid;
        if (nthreads >= TURBO_HEAD_DIM_GPU && e >= TURBO_HEAD_DIM_GPU) {
            continue;
        }
        const int blk = e / bs;
        const int eib = e % bs;
        if constexpr (sizeof(block_t) == sizeof(block_turbo3_0)) {
            smem[e] = dc_codebook_3bit[turbo3_unpack_index((const block_turbo3_0 *)(blocks + ib_chunk + blk), eib)];
        } else {
            smem[e] = dc_codebook_4bit[turbo4_unpack_index((const block_turbo4_0 *)(blocks + ib_chunk + blk), eib)];
        }
    }
    __syncthreads();

    turbo_fwht_smem_128_n<nthreads>(smem, tid);

    const float scale = TURBO_FWHT_SCALE_GPU * __half2float(blocks[ib_chunk].d);
#pragma unroll
    for (int e0 = 0; e0 < TURBO_HEAD_DIM_GPU; e0 += nthreads) {
        const int e = e0 + tid;
        if (nthreads >= TURBO_HEAD_DIM_GPU && e >= TURBO_HEAD_DIM_GPU) {
            continue;
        }
        smem[e] *= scale;
    }
    __syncthreads();
}

// ------------------------------------------------------------
// Chunk decode into registers (one warp per chunk)
// ------------------------------------------------------------
//
// Lane l fills v[0..3] with elements [4l, 4l+4) of the chunk. Equivalent to
// turbo_decode_chunk_smem but without shared memory or block synchronization.

template <typename block_t>
static __device__ __forceinline__ void turbo_decode_chunk_warp(
        const block_t * blocks, int64_t ib_chunk, float v[4], int lane);

template <>
__device__ __forceinline__ void turbo_decode_chunk_warp<block_turbo3_0>(
        const block_turbo3_0 * blocks, int64_t ib_chunk, float v[4], int lane) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int elem          = 4*lane + r;
        const int local_block   = elem / TURBO3_BLOCK_SIZE;
        const int elem_in_block = elem % TURBO3_BLOCK_SIZE;
        v[r] = dc_codebook_3bit[turbo3_unpack_index(blocks + ib_chunk + local_block, elem_in_block)];
    }

    turbo_fwht_warp_128(v, lane);

    const float scale = TURBO_FWHT_SCALE_GPU * __half2float(blocks[ib_chunk].d);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        v[r] *= scale;
    }
}

template <>
__device__ __forceinline__ void turbo_decode_chunk_warp<block_turbo4_0>(
        const block_turbo4_0 * blocks, int64_t ib_chunk, float v[4], int lane) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int elem          = 4*lane + r;
        const int local_block   = elem / TURBO4_BLOCK_SIZE;
        const int elem_in_block = elem % TURBO4_BLOCK_SIZE;
        v[r] = dc_codebook_4bit[turbo4_unpack_index(blocks + ib_chunk + local_block, elem_in_block)];
    }

    turbo_fwht_warp_128(v, lane);

    const float scale = TURBO_FWHT_SCALE_GPU * __half2float(blocks[ib_chunk].d);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        v[r] *= scale;
    }
}
