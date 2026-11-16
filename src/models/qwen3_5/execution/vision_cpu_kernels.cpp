#include "models/qwen3_5/execution/vision_cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_MSC_VER) && defined(NINFER_VISION_CPU_X86)
#    include <intrin.h>
#endif

namespace ninfer::models::qwen3_5::execution::cpu_vision {
namespace {

constexpr std::int32_t kTileRows   = 4;
constexpr std::int32_t kTileTokens = 16;
constexpr std::int32_t kHeadDim    = 72;

void pack_panel(const float* x, std::size_t x_stride, std::int32_t k, std::int32_t token,
                std::int32_t width, float* panel) {
    for (std::int32_t inner = 0; inner < k; ++inner) {
        const float* source = x + std::size_t(inner) * x_stride + token;
        float* target       = panel + std::size_t(inner) * kTileTokens;
        std::copy_n(source, width, target);
        std::fill(target + width, target + kTileTokens, 0.0F);
    }
}

void gemm_packed_bias(const float* w, std::size_t w_stride, const float* bias, std::int32_t n,
                      std::int32_t k, const float* packed, std::int32_t tokens, float* y,
                      std::size_t y_stride) {
    for (std::int32_t token = 0, p = 0; token < tokens; token += kTileTokens, ++p) {
        const std::int32_t width = std::min(kTileTokens, tokens - token);
        const float* panel       = packed + std::size_t(p) * k * kTileTokens;
        for (std::int32_t row = 0; row < n; row += kTileRows) {
            const std::int32_t rows = std::min(kTileRows, n - row);
            float acc[kTileRows][kTileTokens];
            for (std::int32_t r = 0; r < rows; ++r) {
                std::fill_n(acc[r], kTileTokens, bias[row + r]);
            }
            for (std::int32_t inner = 0; inner < k; ++inner) {
                const float* xs = panel + std::size_t(inner) * kTileTokens;
                for (std::int32_t r = 0; r < rows; ++r) {
                    const float weight = w[std::size_t(row + r) * w_stride + inner];
                    for (std::int32_t t = 0; t < kTileTokens; ++t) { acc[r][t] += weight * xs[t]; }
                }
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                std::copy_n(acc[r], width, y + std::size_t(row + r) * y_stride + token);
            }
        }
    }
}

void layer_norm_cols(const float* x_data, std::size_t stride, std::int32_t features,
                     const float* weight, const float* bias, std::int32_t first,
                     std::int32_t count, float epsilon, float* y_data) {
    std::vector<float> mean(std::size_t(count), 0.0F);
    std::vector<float> variance(std::size_t(count), 0.0F);
    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        for (std::int32_t t = 0; t < count; ++t) { mean[t] += xs[t]; }
    }
    for (float& value : mean) { value /= static_cast<float>(features); }
    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        for (std::int32_t t = 0; t < count; ++t) {
            const float centered = xs[t] - mean[t];
            variance[t] += centered * centered;
        }
    }
    for (float& value : variance) {
        value = 1.0F / std::sqrt(value / static_cast<float>(features) + epsilon);
    }
    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        float* ys       = y_data + std::size_t(r) * stride + first;
        for (std::int32_t t = 0; t < count; ++t) {
            ys[t] = (xs[t] - mean[t]) * variance[t] * weight[r] + bias[r];
        }
    }
}

void add_into(float* x, const float* delta, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) { x[i] += delta[i]; }
}

void gelu_tanh(float* x, std::size_t count) {
    constexpr float kScale = 0.7978845608028654F;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = x[i];
        x[i] = 0.5F * value * (1.0F + std::tanh(kScale * (value + 0.044715F * value * value * value)));
    }
}

void rope_pair(float* first, float* second, const float* cosines, const float* sines,
               std::int32_t tokens) {
    for (std::int32_t t = 0; t < tokens; ++t) {
        const float a = first[t];
        const float b = second[t];
        first[t]      = a * cosines[t] - b * sines[t];
        second[t]     = b * cosines[t] + a * sines[t];
    }
}

void segment_attention_block(const float* qkv, std::size_t qkv_stride, std::int32_t hidden,
                             std::int32_t row_base, std::int32_t s0, std::int32_t q0,
                             std::int32_t qn, std::int32_t segment, float scale, float* scores,
                             float* out, std::size_t out_stride) {
    constexpr std::int32_t kKeys = 256;
    for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
        const std::int32_t kn = std::min(kKeys, segment - k0);
        for (std::int32_t q = 0; q < qn; ++q) {
            std::fill_n(scores + std::size_t(q) * segment + k0, kn, 0.0F);
        }
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const float* queries = qkv + std::size_t(row_base + d) * qkv_stride + q0;
            const float* keys    = qkv + std::size_t(hidden + row_base + d) * qkv_stride + s0 + k0;
            for (std::int32_t q = 0; q < qn; ++q) {
                const float coefficient = queries[q] * scale;
                float* line             = scores + std::size_t(q) * segment + k0;
                for (std::int32_t key = 0; key < kn; ++key) { line[key] += coefficient * keys[key]; }
            }
        }
    }
    for (std::int32_t q = 0; q < qn; ++q) {
        float* line   = scores + std::size_t(q) * segment;
        const float m = *std::max_element(line, line + segment);
        float sum     = 0.0F;
        for (std::int32_t key = 0; key < segment; ++key) {
            line[key] = std::exp(line[key] - m);
            sum += line[key];
        }
        const float inverse = 1.0F / sum;
        for (std::int32_t key = 0; key < segment; ++key) { line[key] *= inverse; }
    }
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        std::fill_n(out + std::size_t(row_base + d) * out_stride + q0, qn, 0.0F);
    }
    for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
        const std::int32_t kn = std::min(kKeys, segment - k0);
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const float* values =
                qkv + std::size_t(2 * hidden + row_base + d) * qkv_stride + s0 + k0;
            float* target = out + std::size_t(row_base + d) * out_stride + q0;
            for (std::int32_t q = 0; q < qn; ++q) {
                const float* line = scores + std::size_t(q) * segment + k0;
                float sum         = 0.0F;
                for (std::int32_t key = 0; key < kn; ++key) { sum += line[key] * values[key]; }
                target[q] += sum;
            }
        }
    }
}

bool avx2_fma() noexcept {
#if defined(NINFER_VISION_CPU_X86)
#    if defined(_MSC_VER)
    int basic[4], extended[4];
    __cpuid(basic, 0);
    if (basic[0] < 7) { return false; }
    __cpuid(basic, 1);
    if ((basic[2] & (1 << 12)) == 0 || (basic[2] & (1 << 27)) == 0 ||
        (basic[2] & (1 << 28)) == 0 || (_xgetbv(0) & 6) != 6) {
        return false;
    }
    __cpuidex(extended, 7, 0);
    return (extended[1] & (1 << 5)) != 0;
#    else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#    endif
#else
    return false;
#endif
}

bool avx_supported() noexcept {
#if defined(NINFER_VISION_CPU_X86)
#    if defined(_MSC_VER)
    int basic[4];
    __cpuid(basic, 0);
    if (basic[0] < 1) { return false; }
    __cpuid(basic, 1);
    return (basic[2] & (1 << 27)) != 0 && (basic[2] & (1 << 28)) != 0 && (_xgetbv(0) & 6) == 6;
#    else
    return __builtin_cpu_supports("avx");
#    endif
#else
    return false;
#endif
}

} // namespace

const Kernels& portable_kernels() noexcept {
    static const Kernels kernels{pack_panel, gemm_packed_bias, layer_norm_cols,        add_into,
                                 gelu_tanh,  rope_pair,        segment_attention_block, "portable"};
    return kernels;
}

const Kernels& kernels() noexcept {
#if defined(NINFER_VISION_CPU_X86)
    static const Kernels& chosen = avx2_fma() ? avx2_kernels() :
                                   avx_supported() ? avx_kernels() :
                                   portable_kernels();
    return chosen;
#else
    return portable_kernels();
#endif
}

} // namespace ninfer::models::qwen3_5::execution::cpu_vision
