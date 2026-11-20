#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#include "ops/moe_experts/gguf_expert_cpu.h"  // full GgufCpuMatrix definition

#include "ggml-quants.h"  // NInfer: vendored (block types, quantize_row_*_ref)
#include "quants.h"       // NInfer: vendored (ggml_vec_dot_*)

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops {

// BF16 -> FP32. BF16 shares FP32's 8-bit exponent field (bias 127), so all finite, subnormal,
// and special values decode directly via left shift by 16 bits.
static float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(std::uint32_t(bits) << 16);
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
    : capacity_(token_capacity), k_(k), prepared_(std::size_t(token_capacity), 0) {
    if (capacity_ < 1 || k_ < 32 || k_ % 32 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: bad shape");
    }
}

bool GgmlCpuActivation::all_prepared() const noexcept {
    if (tokens_ <= 0) { return false; }
    for (int t = 0; t < tokens_; ++t) {
        if (prepared_[std::size_t(t)] == 0) { return false; }
    }
    return true;
}

void GgmlCpuActivation::prepare_bf16_start(GgmlActLayout layout, int tokens) {
    if (tokens <= 0 || tokens > capacity_) {
        throw std::invalid_argument("GGUF ggml CPU activation: bad token count");
    }
    if (layout == GgmlActLayout::Q8K && k_ % 256 != 0) {
        throw std::invalid_argument("GGUF ggml CPU activation: k must be a multiple of 256 for Q8_K");
    }
    layout_    = layout;
    row_bytes_ = layout == GgmlActLayout::Q8K ? std::size_t(k_) / 256 * sizeof(block_q8_K)
                                              : std::size_t(k_) / 32 * sizeof(block_q8_0);
    data_.resize(std::size_t(capacity_) * row_bytes_);
    tokens_ = tokens;
    std::fill(prepared_.begin(), prepared_.end(), std::uint8_t(0));
}

// The vendored ggml BF16 quantizers derive block scale/inverse with `_mm256_max_ps` and
// `127 / amax`, which propagates/drops NaN or overflows scale/inverse on extreme finite magnitudes
// (`1e-40` -> Inf `iscale`; `+-1e30` -> Inf FP16 `d` in Q8_0). Validate per block before
// quantizing so both exact and ggml backends reject nonfinite values and scale overflows cleanly.
static void validate_bf16_row(const std::uint16_t* v, int k, GgmlActLayout layout) {
    const int block_size = layout == GgmlActLayout::Q8K ? 256 : 32;
    for (int b = 0; b < k; b += block_size) {
        float amax = 0.0F;
        for (int i = 0; i < block_size; ++i) {
            const std::uint16_t bits = v[b + i];
            if ((bits & 0x7F80U) == 0x7F80U) {
                throw std::runtime_error("GGUF CPU expert: nonfinite activation");
            }
            amax = std::max(amax, std::abs(bf16_to_float(bits)));
        }
        if (amax > 0.0F) {
            const float d = amax / 127.0F;
            if (!std::isfinite(127.0F / amax) || d == 0.0F || !std::isfinite(1.0F / d) ||
                (layout == GgmlActLayout::Q8_0 && d > 65504.0F)) {
                throw std::runtime_error("GGUF CPU expert: nonfinite activation scale");
            }
        }
    }
}

void GgmlCpuActivation::prepare_bf16_token(int token, const std::uint16_t* values) {
    if (token < 0 || token >= tokens_) {
        throw std::out_of_range("GGUF ggml CPU activation: token index out of range");
    }
    validate_bf16_row(values, k_, layout_);
    void* out = data_.data() + std::size_t(token) * row_bytes_;
    if (gguf_ggml_cpu_available()) {
        if (layout_ == GgmlActLayout::Q8K) {
            ggml_quantize_bf16_q8_K(values, out, k_);
        } else {
            ggml_quantize_bf16_q8_0(values, out, k_);
        }
        prepared_[std::size_t(token)] = 1;
        return;
    }
    std::vector<float> local_scratch(static_cast<std::size_t>(k_));
    for (int i = 0; i < k_; ++i) { local_scratch[i] = bf16_to_float(values[i]); }
    if (layout_ == GgmlActLayout::Q8K) {
        quantize_row_q8_K_ref(local_scratch.data(), reinterpret_cast<block_q8_K*>(out), k_);
    } else {
        quantize_row_q8_0_ref(local_scratch.data(), reinterpret_cast<block_q8_0*>(out), k_);
    }
    prepared_[std::size_t(token)] = 1;
}

void GgmlCpuActivation::copy_token_from(int dst_token, const GgmlCpuActivation& src, int src_token) {
    if (dst_token < 0 || dst_token >= tokens_ || src_token < 0 || src_token >= src.tokens_ ||
        src.prepared_[std::size_t(src_token)] == 0 || row_bytes_ != src.row_bytes_) {
        throw std::invalid_argument("GGUF ggml CPU activation: copy of unprepared or mismatched token");
    }
    std::memcpy(data_.data() + std::size_t(dst_token) * row_bytes_, src.codes(src_token), row_bytes_);
    prepared_[std::size_t(dst_token)] = 1;
}

void GgmlCpuActivation::prepare_bf16(GgmlActLayout layout, const std::uint16_t* values,
                                     std::size_t stride, int tokens) {
    prepare_bf16_start(layout, tokens);
    const bool use_simd = gguf_ggml_cpu_available();
    if (!use_simd) { scratch_.resize(k_); }
    for (int t = 0; t < tokens; ++t) {
        const std::uint16_t* v = values + std::size_t(t) * stride;
        validate_bf16_row(v, k_, layout);
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
    std::fill_n(prepared_.begin(), tokens, std::uint8_t(1));
}

const void* GgmlCpuActivation::codes(int token) const noexcept {
    return data_.data() + std::size_t(token) * row_bytes_;
}

std::size_t GgmlCpuActivation::row_bytes() const noexcept { return row_bytes_; }

void gguf_cpu_rows_ggml(const GgufCpuMatrix& w, const GgmlCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride) {
    if (!gguf_ggml_supported(w.format) || w.k != x.k() || !w.data || row_begin < 0 || row_end > w.rows ||
        row_begin > row_end || !x.all_prepared()) {
        throw std::invalid_argument("GGUF ggml CPU expert: mismatched or unprepared activation");
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
