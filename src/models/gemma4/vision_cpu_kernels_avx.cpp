// Built with AVX enabled; reached when the CPU has AVX 1.0 (with or without FMA).

#include "models/gemma4/vision_cpu_kernels.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer::models::gemma4::cpu_vision {
namespace {

constexpr int kTileTokens = 16;

inline __m256 fmadd(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

inline __m256 fnmadd(__m256 a, __m256 b, __m256 c) {
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
}

void pack(const float* x, std::size_t x_stride, int k, int token, int width, float* panel) {
    if (width == kTileTokens) {
        for (int inner = 0; inner < k; ++inner) {
            const float* source = x + std::size_t(inner) * x_stride + token;
            float* target       = panel + std::size_t(inner) * kTileTokens;
            _mm256_storeu_ps(target,     _mm256_loadu_ps(source));
            _mm256_storeu_ps(target + 8, _mm256_loadu_ps(source + 8));
        }
        return;
    }
    for (int inner = 0; inner < k; ++inner) {
        const float* source = x + std::size_t(inner) * x_stride + token;
        float* target       = panel + std::size_t(inner) * kTileTokens;
        std::copy_n(source, width, target);
        std::fill(target + width, target + kTileTokens, 0.0F);
    }
}

inline void store_tile_row(float* dst, int width, __m256 a0, __m256 a1) {
    alignas(32) float out[kTileTokens];
    _mm256_store_ps(out, a0);
    _mm256_store_ps(out + 8, a1);
    if (width == kTileTokens) {
        _mm256_storeu_ps(dst, a0);
        _mm256_storeu_ps(dst + 8, a1);
    } else {
        std::copy_n(out, width, dst);
    }
}

void gemm_panel(const float* w, std::size_t w_stride, int n, int k, const float* panel,
                int width, int token, float* y, std::size_t y_stride) {
    int row = 0;
    // 6x16 microkernel: 12 YMM accumulators hold 6 rows x 16 tokens while streaming 2 XMM/YMM loads.
    for (; row + 6 <= n; row += 6) {
        const float* w0 = w + std::size_t(row) * w_stride;
        const float* w1 = w0 + w_stride;
        const float* w2 = w1 + w_stride;
        const float* w3 = w2 + w_stride;
        const float* w4 = w3 + w_stride;
        const float* w5 = w4 + w_stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        __m256 a40 = _mm256_setzero_ps(), a41 = _mm256_setzero_ps();
        __m256 a50 = _mm256_setzero_ps(), a51 = _mm256_setzero_ps();
        const float* xs = panel;
        for (int inner = 0; inner < k; ++inner, xs += kTileTokens) {
            const __m256 x0 = _mm256_loadu_ps(xs);
            const __m256 x1 = _mm256_loadu_ps(xs + 8);
            __m256 v = _mm256_broadcast_ss(w0 + inner);
            a00      = fmadd(v, x0, a00);
            a01      = fmadd(v, x1, a01);
            v        = _mm256_broadcast_ss(w1 + inner);
            a10      = fmadd(v, x0, a10);
            a11      = fmadd(v, x1, a11);
            v        = _mm256_broadcast_ss(w2 + inner);
            a20      = fmadd(v, x0, a20);
            a21      = fmadd(v, x1, a21);
            v        = _mm256_broadcast_ss(w3 + inner);
            a30      = fmadd(v, x0, a30);
            a31      = fmadd(v, x1, a31);
            v        = _mm256_broadcast_ss(w4 + inner);
            a40      = fmadd(v, x0, a40);
            a41      = fmadd(v, x1, a41);
            v        = _mm256_broadcast_ss(w5 + inner);
            a50      = fmadd(v, x0, a50);
            a51      = fmadd(v, x1, a51);
        }
        float* y0 = y + std::size_t(row) * y_stride + token;
        store_tile_row(y0, width, a00, a01);
        store_tile_row(y0 + y_stride, width, a10, a11);
        store_tile_row(y0 + 2 * y_stride, width, a20, a21);
        store_tile_row(y0 + 3 * y_stride, width, a30, a31);
        store_tile_row(y0 + 4 * y_stride, width, a40, a41);
        store_tile_row(y0 + 5 * y_stride, width, a50, a51);
    }
    if (row + 4 <= n) {
        const float* w0 = w + std::size_t(row) * w_stride;
        const float* w1 = w0 + w_stride;
        const float* w2 = w1 + w_stride;
        const float* w3 = w2 + w_stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        const float* xs = panel;
        for (int inner = 0; inner < k; ++inner, xs += kTileTokens) {
            const __m256 x0 = _mm256_loadu_ps(xs);
            const __m256 x1 = _mm256_loadu_ps(xs + 8);
            __m256 v = _mm256_broadcast_ss(w0 + inner);
            a00 = fmadd(v, x0, a00); a01 = fmadd(v, x1, a01);
            v   = _mm256_broadcast_ss(w1 + inner);
            a10 = fmadd(v, x0, a10); a11 = fmadd(v, x1, a11);
            v   = _mm256_broadcast_ss(w2 + inner);
            a20 = fmadd(v, x0, a20); a21 = fmadd(v, x1, a21);
            v   = _mm256_broadcast_ss(w3 + inner);
            a30 = fmadd(v, x0, a30); a31 = fmadd(v, x1, a31);
        }
        float* y0 = y + std::size_t(row) * y_stride + token;
        store_tile_row(y0, width, a00, a01);
        store_tile_row(y0 + y_stride, width, a10, a11);
        store_tile_row(y0 + 2 * y_stride, width, a20, a21);
        store_tile_row(y0 + 3 * y_stride, width, a30, a31);
        row += 4;
    }
    for (; row + 2 <= n; row += 2) {
        const float* w0 = w + std::size_t(row) * w_stride;
        const float* w1 = w0 + w_stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        const float* xs = panel;
        for (int inner = 0; inner < k; ++inner, xs += kTileTokens) {
            const __m256 x0 = _mm256_loadu_ps(xs);
            const __m256 x1 = _mm256_loadu_ps(xs + 8);
            __m256 v = _mm256_broadcast_ss(w0 + inner);
            a00 = fmadd(v, x0, a00); a01 = fmadd(v, x1, a01);
            v   = _mm256_broadcast_ss(w1 + inner);
            a10 = fmadd(v, x0, a10); a11 = fmadd(v, x1, a11);
        }
        float* y0 = y + std::size_t(row) * y_stride + token;
        store_tile_row(y0, width, a00, a01);
        store_tile_row(y0 + y_stride, width, a10, a11);
    }
    if (row < n) {
        const float* w0 = w + std::size_t(row) * w_stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        const float* xs = panel;
        for (int inner = 0; inner < k; ++inner, xs += kTileTokens) {
            const __m256 x0 = _mm256_loadu_ps(xs);
            const __m256 x1 = _mm256_loadu_ps(xs + 8);
            const __m256 v = _mm256_broadcast_ss(w0 + inner);
            a00 = fmadd(v, x0, a00); a01 = fmadd(v, x1, a01);
        }
        store_tile_row(y + std::size_t(row) * y_stride + token, width, a00, a01);
    }
}

void gemm(const float* w, std::size_t w_stride, int n, int k, const float* x, std::size_t x_stride,
          int tokens, float* y, std::size_t y_stride) {
    thread_local std::vector<float> storage;
    storage.resize(std::size_t(k) * kTileTokens + 16);
    float* panel = reinterpret_cast<float*>(
        (reinterpret_cast<std::uintptr_t>(storage.data()) + 63U) & ~std::uintptr_t(63U));
    for (int token = 0; token < tokens; token += kTileTokens) {
        const int width = std::min(kTileTokens, tokens - token);
        pack(x, x_stride, k, token, width, panel);
        gemm_panel(w, w_stride, n, k, panel, width, token, y, y_stride);
    }
}

void gemm_packed(const float* w, std::size_t w_stride, int n, int k, const float* packed,
                 int tokens, float* y, std::size_t y_stride) {
    for (int token = 0, p = 0; token < tokens; token += kTileTokens, ++p) {
        const int width    = std::min(kTileTokens, tokens - token);
        const float* panel = packed + std::size_t(p) * k * kTileTokens;
        gemm_panel(w, w_stride, n, k, panel, width, token, y, y_stride);
    }
}

// exp(x) for x <= 0: degree-6 polynomial and 2^n in exponent bits using AVX 1.0 + 128-bit integer SSE.
__m256 exp_nonpositive(__m256 x) {
    x              = _mm256_max_ps(x, _mm256_set1_ps(-87.3F));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341F)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = fnmadd(n, _mm256_set1_ps(0.693359375F), x);
    x = fnmadd(n, _mm256_set1_ps(-2.12194440e-4F), x);

    __m256 p = _mm256_set1_ps(1.9875691500e-4F);
    p        = fmadd(p, x, _mm256_set1_ps(1.3981999507e-3F));
    p        = fmadd(p, x, _mm256_set1_ps(8.3334519073e-3F));
    p        = fmadd(p, x, _mm256_set1_ps(4.1665795894e-2F));
    p        = fmadd(p, x, _mm256_set1_ps(1.6666665459e-1F));
    p        = fmadd(p, x, _mm256_set1_ps(5.0000001201e-1F));
    p        = fmadd(p, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0F)));

    const __m128 n_lo  = _mm256_castps256_ps128(n);
    const __m128 n_hi  = _mm256_extractf128_ps(n, 1);
    const __m128i bias = _mm_set1_epi32(127);
    const __m128i sc_lo = _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(n_lo), bias), 23);
    const __m128i sc_hi = _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(n_hi), bias), 23);
    const __m256 scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(sc_lo)),
                                              _mm_castsi128_ps(sc_hi), 1);
    return _mm256_mul_ps(p, scale);
}

void softmax_columns(float* s, std::size_t stride, int rows, int cols, float* sums) {
    thread_local std::vector<float> maxima;
    maxima.assign(std::size_t(cols), -INFINITY);
    const int vector_cols = cols / 8 * 8;
    for (int r = 0; r < rows; ++r) {
        const float* row = s + std::size_t(r) * stride;
        int c            = 0;
        for (; c < vector_cols; c += 8) {
            _mm256_storeu_ps(maxima.data() + c, _mm256_max_ps(_mm256_loadu_ps(maxima.data() + c),
                                                              _mm256_loadu_ps(row + c)));
        }
        for (; c < cols; ++c) { maxima[c] = std::max(maxima[c], row[c]); }
    }
    std::fill_n(sums, cols, 0.0F);
    for (int r = 0; r < rows; ++r) {
        float* row = s + std::size_t(r) * stride;
        int c      = 0;
        for (; c < vector_cols; c += 8) {
            const __m256 e = exp_nonpositive(
                _mm256_sub_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(maxima.data() + c)));
            _mm256_storeu_ps(row + c, e);
            _mm256_storeu_ps(sums + c, _mm256_add_ps(_mm256_loadu_ps(sums + c), e));
        }
        for (; c < cols; ++c) {
            row[c] = std::exp(row[c] - maxima[c]);
            sums[c] += row[c];
        }
    }
}

void geglu(const float* gate, const float* up, float* out, std::size_t count) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0F);
    std::size_t i         = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 x = _mm256_loadu_ps(gate + i);
        const __m256 cube = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        const __m256 inner = _mm256_mul_ps(
            _mm256_set1_ps(0.7978845608028654F),
            fmadd(cube, _mm256_set1_ps(0.044715F), x));
        const __m256 magnitude = _mm256_andnot_ps(sign_bit, inner);
        const __m256 t = exp_nonpositive(_mm256_mul_ps(magnitude, _mm256_set1_ps(-2.0F)));
        const __m256 one = _mm256_set1_ps(1.0F);
        __m256 tanh = _mm256_div_ps(_mm256_sub_ps(one, t), _mm256_add_ps(one, t));
        tanh        = _mm256_or_ps(tanh, _mm256_and_ps(sign_bit, inner));
        const __m256 gelu = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5F), x),
                                          _mm256_add_ps(one, tanh));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(gelu, _mm256_loadu_ps(up + i)));
    }
    for (; i < count; ++i) {
        const float x     = gate[i];
        const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
        out[i]            = 0.5F * x * (1.0F + std::tanh(inner)) * up[i];
    }
}

void rms_accum_sq(const float* row, int count, double* sums) {
    int t = 0;
    for (; t + 8 <= count; t += 8) {
        const __m256 x   = _mm256_loadu_ps(row + t);
        const __m256d d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        const __m256d d1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
        _mm256_storeu_pd(sums + t,
                         _mm256_add_pd(_mm256_loadu_pd(sums + t), _mm256_mul_pd(d0, d0)));
        _mm256_storeu_pd(sums + t + 4,
                         _mm256_add_pd(_mm256_loadu_pd(sums + t + 4), _mm256_mul_pd(d1, d1)));
    }
    for (; t < count; ++t) { sums[t] += double(row[t]) * row[t]; }
}

void scale_mul_row(const float* in, const float* scale, float gamma, int count, float* out) {
    const __m256 vg = _mm256_set1_ps(gamma);
    int t = 0;
    for (; t + 8 <= count; t += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(in + t), _mm256_loadu_ps(scale + t));
        _mm256_storeu_ps(out + t, _mm256_mul_ps(v, vg));
    }
    for (; t < count; ++t) { out[t] = in[t] * scale[t] * gamma; }
}

void add_scaled_row(const float* in, const float* scale, float gamma, int count, float* inout) {
    const __m256 vg = _mm256_set1_ps(gamma);
    int t = 0;
    for (; t + 8 <= count; t += 8) {
        const __m256 d = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + t),
                                                     _mm256_loadu_ps(scale + t)),
                                       vg);
        _mm256_storeu_ps(inout + t, _mm256_add_ps(_mm256_loadu_ps(inout + t), d));
    }
    for (; t < count; ++t) { inout[t] += in[t] * scale[t] * gamma; }
}

} // namespace

const Kernels& avx_kernels() noexcept {
    static const Kernels kernels{gemm,        softmax_columns, geglu,         pack,
                                 gemm_packed, rms_accum_sq,    scale_mul_row, add_scaled_row,
                                 "avx"};
    return kernels;
}

} // namespace ninfer::models::gemma4::cpu_vision
