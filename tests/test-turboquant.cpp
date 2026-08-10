// TurboQuant CPU reference tests: FWHT self-inverse, roundtrip MSE, bit-packing
//
// Validates that the quantize/dequantize pipeline in ggml-quants.c produces
// MSE*d values consistent with the paper (Zandieh et al., ICLR 2026).
//
// head_dim is exercised at 128 and at multiples of it. TurboQuant processes a
// row in fixed 128-element chunks, so a 256-dim head is quantized as two
// independently normalized halves. Each half is a unit vector in R^128, which
// is exactly the distribution the codebooks were fitted for — the reconstruction
// error must therefore land in the same range as for a plain 128-dim head.

#include "ggml.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int CHUNK_DIM  = 128;  // TurboQuant FWHT chunk size (fixed)
static constexpr int BLOCK_SIZE = 32;
static constexpr int N_VECTORS  = 10000;

// head_dim values under test. Every one must be a multiple of CHUNK_DIM.
static constexpr int HEAD_DIMS[]  = { 128, 256, 384 };
static constexpr int N_HEAD_DIMS  = (int)(sizeof(HEAD_DIMS) / sizeof(HEAD_DIMS[0]));

// Expected MSE*d ranges (paper: TQ3 ~0.034, TQ4 ~0.009 for d=128)
static constexpr float TQ3_MSE_D_MIN = 0.025f;
static constexpr float TQ3_MSE_D_MAX = 0.045f;
static constexpr float TQ4_MSE_D_MIN = 0.005f;
static constexpr float TQ4_MSE_D_MAX = 0.015f;

// ============================================================
// FWHT reference (must match ggml-quants.c)
// ============================================================

static void fwht_f32(float * x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

// ============================================================
// Test 1: FWHT self-inverse (FWHT(FWHT(x)) == x), per 128-chunk
// ============================================================

static int test_fwht_self_inverse(int head_dim) {
    printf("  FWHT self-inverse (d=%d, %d chunk(s))... ", head_dim, head_dim / CHUNK_DIM);

    std::vector<float> orig(head_dim);
    std::vector<float> work(head_dim);

    for (int i = 0; i < head_dim; i++) {
        orig[i] = sinf((float)(i + 1) * 0.7f) * 2.0f;
    }
    work = orig;

    // Applied chunk-wise, mirroring how ggml-quants.c walks the row
    for (int off = 0; off < head_dim; off += CHUNK_DIM) {
        fwht_f32(work.data() + off, CHUNK_DIM);
        fwht_f32(work.data() + off, CHUNK_DIM);
    }

    float max_err = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float err = fabsf(work[i] - orig[i]);
        if (err > max_err) { max_err = err; }
    }

    bool pass = max_err < 1e-5f;
    printf("max_err=%.2e %s\n", max_err, pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Test 2 & 3: Roundtrip MSE for TQ3 and TQ4
// ============================================================

static void generate_random_vectors(float * dst, int n_elements, unsigned int seed) {
    unsigned int state = seed;
    for (int i = 0; i < n_elements; i++) {
        state = state * 1664525u + 1013904223u;
        dst[i] = ((float)(state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
    }
}

static int test_roundtrip_mse(ggml_type type, float mse_d_min, float mse_d_max, int head_dim) {
    const char * name = ggml_type_name(type);
    printf("  %s roundtrip MSE*d (n=%d, d=%d)... ", name, N_VECTORS, head_dim);

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    assert(traits->from_float_ref != nullptr);
    assert(traits->to_float != nullptr);

    const int n_elements = N_VECTORS * head_dim;

    std::vector<float> src(n_elements);
    std::vector<float> dst(n_elements);
    size_t quant_size = (size_t)n_elements / BLOCK_SIZE * ggml_type_size(type);
    std::vector<uint8_t> quant(quant_size);

    generate_random_vectors(src.data(), n_elements, 42);

    traits->from_float_ref(src.data(), quant.data(), n_elements);
    traits->to_float(quant.data(), dst.data(), n_elements);

    // MSE*d = E[ ||x - x̃||² / ||x||² ] (normalized reconstruction error)
    double total_nmse = 0.0;
    for (int v = 0; v < N_VECTORS; v++) {
        double err_sq  = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < head_dim; i++) {
            double diff = (double)src[v * head_dim + i] - (double)dst[v * head_dim + i];
            err_sq  += diff * diff;
            norm_sq += (double)src[v * head_dim + i] * (double)src[v * head_dim + i];
        }
        if (norm_sq > 1e-20) {
            total_nmse += err_sq / norm_sq;
        }
    }
    float mse_d = (float)(total_nmse / N_VECTORS);

    bool pass = mse_d >= mse_d_min && mse_d <= mse_d_max;
    printf("MSE*d=%.4f [%.3f..%.3f] %s\n", mse_d, mse_d_min, mse_d_max, pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Test 4: Bit-pack determinism and sanity
// ============================================================

static int test_bitpack_deterministic(ggml_type type, int head_dim) {
    const char * name = ggml_type_name(type);
    printf("  %s pack determinism (d=%d)... ", name, head_dim);

    const ggml_type_traits * traits = ggml_get_type_traits(type);

    std::vector<float> src(head_dim);
    for (int i = 0; i < head_dim; i++) {
        src[i] = cosf((float)i * 0.31415f);
    }

    size_t qsize = (size_t)(head_dim / BLOCK_SIZE) * ggml_type_size(type);
    std::vector<uint8_t> q1(qsize);
    std::vector<uint8_t> q2(qsize);

    traits->from_float_ref(src.data(), q1.data(), head_dim);
    traits->from_float_ref(src.data(), q2.data(), head_dim);

    bool pass = memcmp(q1.data(), q2.data(), qsize) == 0;
    printf("%s\n", pass ? "ok" : "FAILED (non-deterministic)");
    return pass ? 0 : 1;
}

static int test_bitpack_sanity(ggml_type type, int head_dim) {
    const char * name = ggml_type_name(type);
    printf("  %s dequantize sanity (d=%d)... ", name, head_dim);

    const ggml_type_traits * traits = ggml_get_type_traits(type);

    std::vector<float> src(head_dim);
    std::vector<float> dst(head_dim);
    for (int i = 0; i < head_dim; i++) {
        src[i] = cosf((float)i * 0.31415f);
    }

    size_t qsize = (size_t)(head_dim / BLOCK_SIZE) * ggml_type_size(type);
    std::vector<uint8_t> q(qsize);

    traits->from_float_ref(src.data(), q.data(), head_dim);
    traits->to_float(q.data(), dst.data(), head_dim);

    bool all_finite  = true;
    bool any_nonzero = false;
    for (int i = 0; i < head_dim; i++) {
        if (!std::isfinite(dst[i])) { all_finite = false; }
        if (fabsf(dst[i]) > 1e-10f) { any_nonzero = true; }
    }

    bool pass = all_finite && any_nonzero;
    printf("finite=%s nonzero=%s %s\n",
           all_finite ? "yes" : "NO",
           any_nonzero ? "yes" : "NO",
           pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}

// ============================================================
// Test 5: chunk independence
// ============================================================
//
// A 256-dim head must quantize to exactly the same bytes as two separate
// 128-dim heads holding the same values. This is what makes the existing
// codebooks valid for head_dim 256: the halves are processed independently.

static int test_chunk_independence(ggml_type type) {
    const char * name = ggml_type_name(type);
    printf("  %s chunk independence (256 == 2x128)... ", name);

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    const size_t bytes_per_chunk = (size_t)(CHUNK_DIM / BLOCK_SIZE) * ggml_type_size(type);

    std::vector<float> src(2 * CHUNK_DIM);
    generate_random_vectors(src.data(), 2 * CHUNK_DIM, 7);

    // One pass over 256 elements
    std::vector<uint8_t> q_joint(2 * bytes_per_chunk);
    traits->from_float_ref(src.data(), q_joint.data(), 2 * CHUNK_DIM);

    // Two separate passes over 128 elements each
    std::vector<uint8_t> q_split(2 * bytes_per_chunk);
    traits->from_float_ref(src.data(),             q_split.data(),                   CHUNK_DIM);
    traits->from_float_ref(src.data() + CHUNK_DIM, q_split.data() + bytes_per_chunk, CHUNK_DIM);

    bool pass = memcmp(q_joint.data(), q_split.data(), 2 * bytes_per_chunk) == 0;
    printf("%s\n", pass ? "ok" : "FAILED (chunks are not independent)");
    return pass ? 0 : 1;
}

// ============================================================
// Main
// ============================================================

int main(void) {
    printf("TurboQuant CPU reference tests\n");
    printf("==============================\n\n");

    int n_fail  = 0;
    int n_total = 0;

    printf("Test 1: FWHT self-inverse\n");
    for (int i = 0; i < N_HEAD_DIMS; i++) {
        n_fail += test_fwht_self_inverse(HEAD_DIMS[i]);
        n_total++;
    }

    printf("\nTest 2: TQ3 roundtrip MSE\n");
    for (int i = 0; i < N_HEAD_DIMS; i++) {
        n_fail += test_roundtrip_mse(GGML_TYPE_TURBO3_0, TQ3_MSE_D_MIN, TQ3_MSE_D_MAX, HEAD_DIMS[i]);
        n_total++;
    }

    printf("\nTest 3: TQ4 roundtrip MSE\n");
    for (int i = 0; i < N_HEAD_DIMS; i++) {
        n_fail += test_roundtrip_mse(GGML_TYPE_TURBO4_0, TQ4_MSE_D_MIN, TQ4_MSE_D_MAX, HEAD_DIMS[i]);
        n_total++;
    }

    printf("\nTest 4: Bit-pack tests\n");
    for (int i = 0; i < N_HEAD_DIMS; i++) {
        n_fail += test_bitpack_deterministic(GGML_TYPE_TURBO3_0, HEAD_DIMS[i]);
        n_fail += test_bitpack_deterministic(GGML_TYPE_TURBO4_0, HEAD_DIMS[i]);
        n_fail += test_bitpack_sanity(GGML_TYPE_TURBO3_0, HEAD_DIMS[i]);
        n_fail += test_bitpack_sanity(GGML_TYPE_TURBO4_0, HEAD_DIMS[i]);
        n_total += 4;
    }

    printf("\nTest 5: Chunk independence\n");
    n_fail += test_chunk_independence(GGML_TYPE_TURBO3_0);
    n_fail += test_chunk_independence(GGML_TYPE_TURBO4_0);
    n_total += 2;

    printf("\n==============================\n");
    printf("%d/%d tests passed\n", n_total - n_fail, n_total);

    return n_fail > 0 ? 1 : 0;
}
