#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo.cuh"

// ============================================================
// Fused flash-attention vector kernel for TurboQuant KV
// ============================================================
//
// The generic vector kernel cannot handle turbo: it decodes KV element-wise,
// while a TurboQuant value needs the inverse FWHT over its whole 128-element
// chunk. Decoding every KV token would cost O(n_kv) transforms and be slower
// than the bulk conversion this is meant to replace.
//
// Two properties of the Hadamard matrix H (symmetric, orthogonal) avoid that.
// With s = 1/sqrt(128), a stored chunk decodes as  x = norm * s * H c:
//
//   K side:  <x_j, q> = norm_j * s * <H c_j, q> = norm_j * s * <c_j, H q>
//            Transform Q once at kernel start, then dot against raw codebook
//            values — no transform in the inner loop.
//
//   V side:  sum_j p_j x_j = H ( sum_j p_j norm_j s c_j )
//            Accumulate in the Hadamard domain, transform once at the end.
//            Valid because the online-softmax rescaling is a scalar factor and
//            H is linear: H(a*v) = a*H(v).
//
// Two transforms per launch instead of two per KV token.
//
// Layout: one warp owns a full 128-element chunk, lane l holding elements
// [4l, 4l+4). That is exactly the register layout turbo_fwht_warp_128 expects,
// so the transforms need no shared memory.
//
// The accumulator is float, not half: codebook values are small (|c| <= 0.24)
// and the final transform amplifies rounding by sqrt(128).

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

static constexpr __device__ int ggml_cuda_fattn_vec_turbo_nthreads() {
    return 128;
}

template<int D, int ncols, ggml_type type_KV, bool use_logit_softcap>
__launch_bounds__(ggml_cuda_fattn_vec_turbo_nthreads(), 1)
static __global__ void flash_attn_ext_vec_turbo(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    using block_t = typename std::conditional<type_KV == GGML_TYPE_TURBO3_0,
                                              block_turbo3_0, block_turbo4_0>::type;

    static_assert(D % TURBO_HEAD_DIM_GPU == 0, "TurboQuant needs head_dim to be a multiple of 128");
    constexpr int nchunks  = D / TURBO_HEAD_DIM_GPU;   // 1 for D=128, 2 for D=256
    constexpr int nthreads = ggml_cuda_fattn_vec_turbo_nthreads();
    constexpr int nwarps   = nthreads / WARP_SIZE;

    const int lane = threadIdx.x;                      // 0..31, owns elements [4*lane, 4*lane+4)
    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;

    const int ic0 = blockIdx.x * ncols;

    const int sequence  = blockIdx.z / ne02;
    const int head      = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    Q += nb03*sequence + nb02*head + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh = (const half *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Shared memory: KQ logits during the loop, warp combination at the end.
    constexpr int ne_KQ      = ncols*nthreads;
    constexpr int ne_combine = nwarps*D;
    __shared__ float KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];

    // --- Q into the Hadamard domain, once per launch ---
    // Every warp keeps its own copy; that costs a few registers but avoids a
    // block-wide barrier in the hot loop.
    float Q_h[ncols][nchunks][4];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float * Q_j = (const float *) (Q + j*nb01);
#pragma unroll
        for (int c = 0; c < nchunks; ++c) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = c*TURBO_HEAD_DIM_GPU + 4*lane + r;
                Q_h[j][c][r] = (ncols == 1 || ic0 + j < int(ne01.z)) ? Q_j[i] : 0.0f;
            }
            turbo_fwht_warp_128(Q_h[j][c], lane);
            // Fold in the attention scale and the decode-side 1/sqrt(128) here so
            // the inner loop only multiplies by the per-chunk norm.
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                Q_h[j][c][r] *= scale * TURBO_FWHT_SCALE_GPU;
            }
        }
    }

    // VKQ accumulator, kept in the Hadamard domain until the very end.
    float VKQ[ncols][nchunks][4] = {};
    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // --- logits: warp y handles KV tokens [y*WARP_SIZE, (y+1)*WARP_SIZE) ---
        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < WARP_SIZE; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + i_KQ_0;

            const block_t * Kb = (const block_t *) (K + i_KQ*nb11);

            // Codebook dot in the Hadamard domain, per chunk.
            float sum[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sum[j] = 0.0f;
            }
#pragma unroll
            for (int c = 0; c < nchunks; ++c) {
                const int64_t ib = (int64_t)c * TURBO_BLOCKS_PER_CHUNK_GPU;
                float cb[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int elem = 4*lane + r;
                    const int blk  = elem / 32;
                    const int eib  = elem % 32;
                    if constexpr (type_KV == GGML_TYPE_TURBO3_0) {
                        cb[r] = dc_codebook_3bit[turbo3_unpack_index(Kb + ib + blk, eib)];
                    } else {
                        cb[r] = dc_codebook_4bit[turbo4_unpack_index(Kb + ib + blk, eib)];
                    }
                }
                const float norm = __half2float(Kb[ib].d);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    float partial = 0.0f;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        partial += cb[r] * Q_h[j][c][r];
                    }
                    sum[j] += partial * norm;
                }
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sum[j] = warp_reduce_sum<WARP_SIZE>(sum[j]);

                if (use_logit_softcap) {
                    sum[j] = logit_softcap * tanhf(sum[j]);
                }
                // maskh already advances with the loop, so index by i_KQ only, and
                // the row stride is ne11 (the KV length), not ne31.
                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum[j] += slope * __half2float(maskh[j*ne11 + i_KQ]);
                }

                // The offset keeps the running maximum finite when a whole tile is
                // masked out; without it expf(-inf - -inf) yields NaN.
                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum[j] + FATTN_KQ_MAX_OFFSET);

                if (uint32_t(i_KQ_0) == threadIdx.x) {   // one lane keeps each token's logit
                    KQ_reg[j] = sum[j];
                }
            }
        }

        // --- online softmax rescale ---
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = warp_reduce_max<WARP_SIZE>(KQ_max_new[j]);
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

            // Rescaling in the Hadamard domain is exact: H(a*v) = a*H(v).
#pragma unroll
            for (int c = 0; c < nchunks; ++c) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    VKQ[j][c][r] *= KQ_max_scale;
                }
            }
        }

        __syncthreads();

        // --- V accumulation, no transform in the loop ---
#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; ++k0) {
            const int k = threadIdx.y*WARP_SIZE + k0;

            const block_t * Vb = (const block_t *) (V + k*nb21);

#pragma unroll
            for (int c = 0; c < nchunks; ++c) {
                const int64_t ib = (int64_t)c * TURBO_BLOCKS_PER_CHUNK_GPU;
                float cb[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int elem = 4*lane + r;
                    const int blk  = elem / 32;
                    const int eib  = elem % 32;
                    if constexpr (type_KV == GGML_TYPE_TURBO3_0) {
                        cb[r] = dc_codebook_3bit[turbo3_unpack_index(Vb + ib + blk, eib)];
                    } else {
                        cb[r] = dc_codebook_4bit[turbo4_unpack_index(Vb + ib + blk, eib)];
                    }
                }
                const float vnorm = TURBO_FWHT_SCALE_GPU * __half2float(Vb[ib].d);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const float p = KQ[j*nthreads + threadIdx.y*WARP_SIZE + k0] * vnorm;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        VKQ[j][c][r] += p * cb[r];
                    }
                }
            }
        }

        __syncthreads();
    }

    // --- attention sinks ---
    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float kqmax_new    = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new);
            KQ_max[j] = kqmax_new;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (tid == 0 ? expf(sink - KQ_max[j]) : 0.0f);
#pragma unroll
            for (int c = 0; c < nchunks; ++c) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    VKQ[j][c][r] *= KQ_max_scale;
                }
            }
        }
    }

    // --- leave the Hadamard domain: one transform per chunk, per warp ---
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int c = 0; c < nchunks; ++c) {
            turbo_fwht_warp_128(VKQ[j][c], lane);
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        // Stage this warp's contribution for the cross-warp sum.
        float * VKQ_tmp = KQ + threadIdx.y*D;
#pragma unroll
        for (int c = 0; c < nchunks; ++c) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                VKQ_tmp[c*TURBO_HEAD_DIM_GPU + 4*lane + r] = VKQ[j_VKQ][c][r] * kqmax_scale;
            }
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0.0f;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
                    dst_val += KQ[w*D + i0 + tid];
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }
    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] =
            make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_KV, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_turbo_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nthreads = 128;
    constexpr int nwarps   = nthreads / WARP_SIZE;

    fattn_kernel_t fattn_kernel = flash_attn_ext_vec_turbo<D, cols_per_block, type_KV, use_logit_softcap>;

    // No conversion: the kernel reads turbo blocks directly. That is the point.
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, /*nbytes_shared*/ 0,
                                       D, /*need_f16_K*/ false, /*need_f16_V*/ false, /*stream_k*/ false);
}

template <int D, ggml_type type_KV>
void ggml_cuda_flash_attn_ext_vec_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cols_per_block = Q->ne[1] > 1 ? 2 : 1;

    if (logit_softcap == 0.0f) {
        if (cols_per_block == 1) {
            ggml_cuda_flash_attn_ext_vec_turbo_case_impl<D, 1, type_KV, false>(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_turbo_case_impl<D, 2, type_KV, false>(ctx, dst);
        }
    } else {
        if (cols_per_block == 1) {
            ggml_cuda_flash_attn_ext_vec_turbo_case_impl<D, 1, type_KV, true>(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_turbo_case_impl<D, 2, type_KV, true>(ctx, dst);
        }
    }
}
