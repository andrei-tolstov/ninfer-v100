#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#include "ops/moe_experts/gguf_expert_cpu.h"  // full GgufCpuMatrix definition

#include "ggml-quants.h"  // NInfer: vendored (block types, quantize_row_*_ref)
#include "quants.h"       // NInfer: vendored (ggml_vec_dot_*)

#include <bit>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {

// BF16 -> FP32. A BF16 value's 1/8/7 bit fields sit exactly in FP32's 1/8/23, so the
// exponent-normal range is a plain left shift; BF16 subnormals (below 2^-126) are not
// bit-shiftable, so they decode through their value (2^-133 per mantissa unit).
static float bf16_to_float(std::uint16_t bits) noexcept {
    if ((bits & 0x7F80U) == 0) {  // subnormal or zero
        if (bits == 0) { return 0.0f; }
        return static_cast<float>(bits & 0x7FU) * 1.1754943508e-40f;  // 2^-133
    }
    std::uint32_t v = std::uint32_t(bits) << 16;
    return std::bit_cast<float>(v);
}

bool gguf_ggml_supported(QType format) noexcept {
    switch (format) {
    case QType::GGUF_IQ2_XXS:
    case QType::GGUF_IQ2_XS:
    case QType::GGUF_IQ2_S:
    case QType::GGUF_IQ3_XXS:
    case QType::GGUF_IQ3_S:
    case QType::GGUF_IQ4_XS:
    case QType::GGUF_IQ4_NL:
    case QType::GGUF_Q8_0: return true;
    default: return false;
    }
}

#ifndef NINFER_GGML_CPU_REQUIRES_AVX2
#    define NINFER_GGML_CPU_REQUIRES_AVX2 1
#endif

bool gguf_ggml_cpu_available() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    // ggml-cpu's x86 kernels compile for one ISA (NINFER_GGML_CPU_AVX2); require the same
    // instruction sets as the exact backends of that ISA.
    return gguf_cpu_backend_available(NINFER_GGML_CPU_REQUIRES_AVX2 ? GgufCpuBackend::Avx2
                                                                    : GgufCpuBackend::Avx);
#else
    // Only arch/x86/quants.c is vendored; elsewhere ggml_vec_dot_* have no definition
    // (arch-fallback.h maps generic kernels only for GGML_CPU_GENERIC and non-ARM targets).
    return false;
#endif
}

GgmlCpuActivation::GgmlCpuActivation(int token_capacity, int k)
    : capacity_(token_capacity), k_(k) {
    if (capacity_ < 1 || k_ < 32 || k_ % 32 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: bad shape");
    }
}

void GgmlCpuActivation::prepare_bf16_start(GgmlActLayout layout, int tokens) {
    if (tokens > capacity_) { throw std::invalid_argument("GGUF ggml CPU activation: too many tokens"); }
    if (layout == GgmlActLayout::Q8K && k_ % 256 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: k must be a multiple of 256 for Q8_K");
    }
    layout_    = layout;
    row_bytes_ = layout == GgmlActLayout::Q8K ? std::size_t(k_) / 256 * sizeof(block_q8_K)
                                              : std::size_t(k_) / 32 * sizeof(block_q8_0);
    data_.resize(std::size_t(capacity_) * row_bytes_);
    tokens_ = tokens;
}

void GgmlCpuActivation::prepare_bf16_token(int token, const std::uint16_t* values) {
    void* out = data_.data() + std::size_t(token) * row_bytes_;
    if (gguf_ggml_cpu_available()) {
        if (layout_ == GgmlActLayout::Q8K) {
            ggml_quantize_bf16_q8_K(values, out, k_);
        } else {
            ggml_quantize_bf16_q8_0(values, out, k_);
        }
        return;
    }
    std::vector<float> local_scratch(static_cast<std::size_t>(k_));
    for (int i = 0; i < k_; ++i) { local_scratch[i] = bf16_to_float(values[i]); }
    if (layout_ == GgmlActLayout::Q8K) {
        quantize_row_q8_K_ref(local_scratch.data(), reinterpret_cast<block_q8_K*>(out), k_);
    } else {
        quantize_row_q8_0_ref(local_scratch.data(), reinterpret_cast<block_q8_0*>(out), k_);
    }
}

void GgmlCpuActivation::copy_token_from(int dst_token, const GgmlCpuActivation& src, int src_token) {
    std::memcpy(data_.data() + std::size_t(dst_token) * row_bytes_, src.codes(src_token), row_bytes_);
}

void GgmlCpuActivation::prepare_bf16(GgmlActLayout layout, const std::uint16_t* values,
                                     std::size_t stride, int tokens) {
    prepare_bf16_start(layout, tokens);
    const bool use_simd = gguf_ggml_cpu_available();
    if (!use_simd) { scratch_.resize(k_); }
    for (int t = 0; t < tokens; ++t) {
        const std::uint16_t* v = values + std::size_t(t) * stride;
        void* out = data_.data() + std::size_t(t) * row_bytes_;
        if (use_simd) {
            if (layout == GgmlActLayout::Q8K) {
                ggml_quantize_bf16_q8_K(v, out, k_);
            } else {
                ggml_quantize_bf16_q8_0(v, out, k_);
            }
        } else {
            float* x = scratch_.data();
            for (int i = 0; i < k_; ++i) { x[i] = bf16_to_float(v[i]); }
            if (layout == GgmlActLayout::Q8K) {
                quantize_row_q8_K_ref(x, reinterpret_cast<block_q8_K*>(out), k_);
            } else {
                quantize_row_q8_0_ref(x, reinterpret_cast<block_q8_0*>(out), k_);
            }
        }
    }
}

const void* GgmlCpuActivation::codes(int token) const noexcept {
    return data_.data() + std::size_t(token) * row_bytes_;
}

std::size_t GgmlCpuActivation::row_bytes() const noexcept { return row_bytes_; }

void gguf_cpu_rows_ggml(const GgufCpuMatrix& w, const GgmlCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride) {
    if (!gguf_ggml_supported(w.format) || w.k != x.k() || !w.data || row_begin < 0 || row_end > w.rows ||
        row_begin > row_end || x.tokens() == 0) {
        throw std::invalid_argument("GGUF ggml CPU expert: mismatched matrix, activation or rows");
    }
    const int rows = row_end - row_begin;
    const int ntok = x.tokens();
    const void* vx = w.data + std::size_t(row_begin) * w.row_bytes;
    const void* vy = x.codes(0);
    const std::size_t by = x.row_bytes();
    float* out = y + row_begin;
    switch (w.format) {
    case QType::GGUF_IQ2_XXS:
        ggml_vec_dot_iq2_xxs_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ2_XS:
        ggml_vec_dot_iq2_xs_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ2_S:
        ggml_vec_dot_iq2_s_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ3_XXS:
        ggml_vec_dot_iq3_xxs_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ3_S:
        ggml_vec_dot_iq3_s_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ4_XS:
        ggml_vec_dot_iq4_xs_q8_K_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_IQ4_NL:
        ggml_vec_dot_iq4_nl_q8_0_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    case QType::GGUF_Q8_0:
        ggml_vec_dot_q8_0_q8_0_rows(w.k, rows, ntok, out, y_stride, vx, w.row_bytes, vy, by);
        break;
    default:
        throw std::invalid_argument("GGUF ggml CPU expert: unsupported format");
    }
}

}  // namespace ninfer::ops
