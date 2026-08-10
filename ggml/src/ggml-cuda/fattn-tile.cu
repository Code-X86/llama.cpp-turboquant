#include "common.cuh"
#include "fattn-tile.cuh"
#include "fattn-wmma-f16.cuh"

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // TurboQuant V is decoded inside the kernel rather than expanded to f16 by
    // launch_fattn. Only V matters here: the K tile holds nbatch_K < 128 columns
    // at these head sizes, which is less than one FWHT chunk, so K still goes
    // through the conversion path.
    // Only the homogeneous case is instantiated; mixed turbo/f16 keeps using the
    // conversion path, which stays correct because launch_fattn handles it.
    if ((V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0) && K->type == V->type) {
        GGML_ASSERT(V->ne[0] == K->ne[0]);
        switch (K->ne[0]) {
            case 128:
                if (V->type == GGML_TYPE_TURBO3_0) {
                    ggml_cuda_flash_attn_ext_tile_case<128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_tile_case<128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst);
                }
                return;
            case 256:
                if (V->type == GGML_TYPE_TURBO3_0) {
                    ggml_cuda_flash_attn_ext_tile_case<256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_tile_case<256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst);
                }
                return;
            default:
                GGML_ABORT("unsupported head size for turbo V");
        }
    }

    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
