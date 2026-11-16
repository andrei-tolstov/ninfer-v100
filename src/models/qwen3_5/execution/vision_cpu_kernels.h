#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::execution::cpu_vision {

struct Kernels {
    void (*pack_panel)(const float* x, std::size_t x_stride, std::int32_t k, std::int32_t token,
                       std::int32_t width, float* panel);
    void (*gemm_packed_bias)(const float* w, std::size_t w_stride, const float* bias,
                             std::int32_t n, std::int32_t k, const float* packed,
                             std::int32_t tokens, float* y, std::size_t y_stride);
    void (*layer_norm_cols)(const float* x_data, std::size_t stride, std::int32_t features,
                            const float* weight, const float* bias, std::int32_t first,
                            std::int32_t count, float epsilon, float* y_data);
    void (*add_into)(float* x, const float* delta, std::size_t count);
    void (*gelu_tanh)(float* x, std::size_t count);
    void (*rope_pair)(float* first, float* second, const float* cosines, const float* sines,
                      std::int32_t tokens);
    void (*segment_attention_block)(const float* qkv, std::size_t qkv_stride, std::int32_t hidden,
                                    std::int32_t row_base, std::int32_t s0, std::int32_t q0,
                                    std::int32_t qn, std::int32_t segment, float scale,
                                    float* scores, float* out, std::size_t out_stride);
    const char* name;
};

[[nodiscard]] const Kernels& portable_kernels() noexcept;
#if defined(NINFER_VISION_CPU_X86)
[[nodiscard]] const Kernels& avx_kernels() noexcept;
[[nodiscard]] const Kernels& avx2_kernels() noexcept;
#endif
[[nodiscard]] const Kernels& kernels() noexcept;

} // namespace ninfer::models::qwen3_5::execution::cpu_vision
