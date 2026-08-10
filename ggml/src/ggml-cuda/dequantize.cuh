#pragma once

#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// ============================================================
// TurboQuant GPU dequantize device functions
// ============================================================

__device__ __constant__ static float dc_codebook_3bit[8] = {
    -0.1883972972f, -0.1181399059f, -0.0665857641f, -0.0216044751f,
     0.0216041461f,  0.0665854520f,  0.1181396281f,  0.1883970748f
};

__device__ __constant__ static float dc_codebook_4bit[16] = {
    -0.2376389871f, -0.1808080141f, -0.1417777640f, -0.1102646123f,
    -0.0828112376f, -0.0577640422f, -0.0341540905f, -0.0113168380f,
     0.0112761586f,  0.0341139667f,  0.0577250301f,  0.0827738972f,
     0.1102295202f,  0.1417455465f,  0.1807794468f,  0.2376153882f
};

static __device__ __forceinline__ void dequantize_turbo3_0(
    const void * vx, const int64_t ib, const int iqs, float2 & v)
{
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx + ib;
    const uint8_t * qs = x->qs;

    // Unpack two consecutive 3-bit indices
    int elem0 = iqs * 2;
    int elem1 = iqs * 2 + 1;

    // Extract 3-bit value for elem0
    int bit_off0 = elem0 * 3;
    int byte0 = bit_off0 / 8;
    int shift0 = bit_off0 % 8;
    uint16_t raw0 = (uint16_t)qs[byte0] >> shift0;
    if (shift0 > 5 && byte0 + 1 < 12)
        raw0 |= (uint16_t)qs[byte0 + 1] << (8 - shift0);
    uint8_t idx0 = (uint8_t)(raw0 & 0x07);

    // Extract 3-bit value for elem1
    int bit_off1 = elem1 * 3;
    int byte1 = bit_off1 / 8;
    int shift1 = bit_off1 % 8;
    uint16_t raw1 = (uint16_t)qs[byte1] >> shift1;
    if (shift1 > 5 && byte1 + 1 < 12)
        raw1 |= (uint16_t)qs[byte1 + 1] << (8 - shift1);
    uint8_t idx1 = (uint8_t)(raw1 & 0x07);

    const float norm = __half2float(x->d);
    v.x = dc_codebook_3bit[idx0] * norm;
    v.y = dc_codebook_3bit[idx1] * norm;
}

static __device__ __forceinline__ void dequantize_turbo4_0(
    const void * vx, const int64_t ib, const int iqs, float2 & v)
{
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx + ib;

    // 4-bit: 2 values per byte, simple nibble extraction
    uint8_t packed = x->qs[iqs];
    uint8_t idx0 = packed & 0x0F;
    uint8_t idx1 = (packed >> 4) & 0x0F;

    const float norm = __half2float(x->d);
    v.x = dc_codebook_4bit[idx0] * norm;
    v.y = dc_codebook_4bit[idx1] * norm;
}
