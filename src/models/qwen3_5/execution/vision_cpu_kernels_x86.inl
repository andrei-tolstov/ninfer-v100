#include "models/qwen3_5/execution/vision_cpu_kernels.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution::cpu_vision {
namespace {

constexpr std::int32_t kTileTokens = 16;

inline __m256 fmadd(__m256 a, __m256 b, __m256 c) {
#if defined(NINFER_QWEN_VISION_AVX2)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

inline __m256 fnmadd(__m256 a, __m256 b, __m256 c) {
#if defined(NINFER_QWEN_VISION_AVX2)
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

inline float hsum_f32_8(__m256 x) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
}

inline __m256 max_nan_ps(__m256 a, __m256 b) {
    const __m256 m = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(m, a, _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
}

inline __m128 max_nan_ps128(__m128 a, __m128 b) {
    const __m128 nan_a = _mm_cmpunord_ps(a, a);
    const __m128 m     = _mm_max_ps(a, b);
    return _mm_or_ps(_mm_and_ps(nan_a, a), _mm_andnot_ps(nan_a, m));
}

inline float hmax_f32_8(__m256 x) {
    const __m128 s = max_nan_ps128(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 h = max_nan_ps128(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(max_nan_ps128(h, _mm_shuffle_ps(h, h, 1)));
}

__m256 exp_nonpositive(__m256 x) {
    // Keep -87.3F as the first operand: VMAXPS returns the second operand when either is NaN, so
    // NaN propagates to the caller instead of turning into exp(-87.3F).
    x              = _mm256_max_ps(_mm256_set1_ps(-87.3F), x);
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

#if defined(NINFER_QWEN_VISION_AVX2)
    const __m256i scale = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(scale));
#else
    const __m128i bias = _mm_set1_epi32(127);
    const __m128i sc0  = _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_castps256_ps128(n)), bias), 23);
    const __m128i sc1  = _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm256_extractf128_ps(n, 1)), bias), 23);
    const __m256 scale = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(sc0)),
                                              _mm_castsi128_ps(sc1), 1);
    return _mm256_mul_ps(p, scale);
#endif
}

float exp_nonpositive_scalar(float x) {
    return _mm_cvtss_f32(_mm256_castps256_ps128(exp_nonpositive(_mm256_set1_ps(x))));
}

void pack_panel(const float* x, std::size_t x_stride, std::int32_t k, std::int32_t token,
                std::int32_t width, float* panel) {
    if (k <= 0 || width <= 0) { return; }
    if (width == kTileTokens) {
        for (std::int32_t inner = 0; inner < k; ++inner) {
            const float* source = x + std::size_t(inner) * x_stride + token;
            float* target       = panel + std::size_t(inner) * kTileTokens;
            _mm256_storeu_ps(target,     _mm256_loadu_ps(source));
            _mm256_storeu_ps(target + 8, _mm256_loadu_ps(source + 8));
        }
        return;
    }
    for (std::int32_t inner = 0; inner < k; ++inner) {
        const float* source = x + std::size_t(inner) * x_stride + token;
        float* target       = panel + std::size_t(inner) * kTileTokens;
        std::copy_n(source, width, target);
        std::fill(target + width, target + kTileTokens, 0.0F);
    }
}

inline void store_tile_row(float* dst, std::int32_t width, __m256 a0, __m256 a1) {
    if (width == kTileTokens) {
        _mm256_storeu_ps(dst, a0);
        _mm256_storeu_ps(dst + 8, a1);
    } else {
        alignas(32) float out[kTileTokens];
        _mm256_store_ps(out, a0);
        _mm256_store_ps(out + 8, a1);
        std::copy_n(out, width, dst);
    }
}

void gemm_packed_bias(const float* w, std::size_t w_stride, const float* bias, std::int32_t n,
                      std::int32_t k, const float* packed, std::int32_t tokens, float* y,
                      std::size_t y_stride) {
    if (n <= 0 || k <= 0 || tokens <= 0) { return; }
    for (std::int32_t token = 0, p = 0; token < tokens; token += kTileTokens, ++p) {
        const std::int32_t width = std::min(kTileTokens, tokens - token);
        const float* panel       = packed + std::size_t(p) * k * kTileTokens;
        std::int32_t row         = 0;
        for (; row + 6 <= n; row += 6) {
            const float* w0 = w + std::size_t(row) * w_stride;
            const float* w1 = w0 + w_stride;
            const float* w2 = w1 + w_stride;
            const float* w3 = w2 + w_stride;
            const float* w4 = w3 + w_stride;
            const float* w5 = w4 + w_stride;
            __m256 a00 = _mm256_broadcast_ss(bias + row),     a01 = a00;
            __m256 a10 = _mm256_broadcast_ss(bias + row + 1), a11 = a10;
            __m256 a20 = _mm256_broadcast_ss(bias + row + 2), a21 = a20;
            __m256 a30 = _mm256_broadcast_ss(bias + row + 3), a31 = a30;
            __m256 a40 = _mm256_broadcast_ss(bias + row + 4), a41 = a40;
            __m256 a50 = _mm256_broadcast_ss(bias + row + 5), a51 = a50;
            const float* xs = panel;
            for (std::int32_t inner = 0; inner < k; ++inner, xs += kTileTokens) {
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
                v   = _mm256_broadcast_ss(w4 + inner);
                a40 = fmadd(v, x0, a40); a41 = fmadd(v, x1, a41);
                v   = _mm256_broadcast_ss(w5 + inner);
                a50 = fmadd(v, x0, a50); a51 = fmadd(v, x1, a51);
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
            __m256 a00 = _mm256_broadcast_ss(bias + row),     a01 = a00;
            __m256 a10 = _mm256_broadcast_ss(bias + row + 1), a11 = a10;
            __m256 a20 = _mm256_broadcast_ss(bias + row + 2), a21 = a20;
            __m256 a30 = _mm256_broadcast_ss(bias + row + 3), a31 = a30;
            const float* xs = panel;
            for (std::int32_t inner = 0; inner < k; ++inner, xs += kTileTokens) {
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
            __m256 a00 = _mm256_broadcast_ss(bias + row),     a01 = a00;
            __m256 a10 = _mm256_broadcast_ss(bias + row + 1), a11 = a10;
            const float* xs = panel;
            for (std::int32_t inner = 0; inner < k; ++inner, xs += kTileTokens) {
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
            __m256 a00 = _mm256_broadcast_ss(bias + row), a01 = a00;
            const float* xs = panel;
            for (std::int32_t inner = 0; inner < k; ++inner, xs += kTileTokens) {
                const __m256 x0 = _mm256_loadu_ps(xs);
                const __m256 x1 = _mm256_loadu_ps(xs + 8);
                const __m256 v = _mm256_broadcast_ss(w0 + inner);
                a00 = fmadd(v, x0, a00); a01 = fmadd(v, x1, a01);
            }
            store_tile_row(y + std::size_t(row) * y_stride + token, width, a00, a01);
        }
    }
}

void layer_norm_cols(const float* x_data, std::size_t stride, std::int32_t features,
                     const float* weight, const float* bias, std::int32_t first,
                     std::int32_t count, float epsilon, float* y_data) {
    if (features <= 0 || count <= 0) { return; }
    thread_local std::vector<float> mean;
    thread_local std::vector<float> variance;
    mean.assign(std::size_t(count), 0.0F);
    variance.assign(std::size_t(count), 0.0F);
    const std::int32_t vec_end = count / 8 * 8;

    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        std::int32_t t  = 0;
        for (; t < vec_end; t += 8) {
            _mm256_storeu_ps(mean.data() + t,
                             _mm256_add_ps(_mm256_loadu_ps(mean.data() + t), _mm256_loadu_ps(xs + t)));
        }
        for (; t < count; ++t) { mean[t] += xs[t]; }
    }
    const float inv_f   = 1.0F / static_cast<float>(features);
    const __m256 vfeat  = _mm256_set1_ps(static_cast<float>(features));
    for (std::int32_t t = 0; t < vec_end; t += 8) {
        _mm256_storeu_ps(mean.data() + t, _mm256_div_ps(_mm256_loadu_ps(mean.data() + t), vfeat));
    }
    for (std::int32_t t = vec_end; t < count; ++t) { mean[t] /= static_cast<float>(features); }
    (void)inv_f;

    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        std::int32_t t  = 0;
        for (; t < vec_end; t += 8) {
            const __m256 c = _mm256_sub_ps(_mm256_loadu_ps(xs + t), _mm256_loadu_ps(mean.data() + t));
            _mm256_storeu_ps(variance.data() + t,
                             _mm256_add_ps(_mm256_loadu_ps(variance.data() + t), _mm256_mul_ps(c, c)));
        }
        for (; t < count; ++t) {
            const float c = xs[t] - mean[t];
            variance[t] += c * c;
        }
    }
    const __m256 vone = _mm256_set1_ps(1.0F);
    const __m256 veps = _mm256_set1_ps(epsilon);
    for (std::int32_t t = 0; t < vec_end; t += 8) {
        const __m256 var = _mm256_add_ps(_mm256_div_ps(_mm256_loadu_ps(variance.data() + t), vfeat), veps);
        _mm256_storeu_ps(variance.data() + t, _mm256_div_ps(vone, _mm256_sqrt_ps(var)));
    }
    for (std::int32_t t = vec_end; t < count; ++t) {
        variance[t] = 1.0F / std::sqrt(variance[t] / static_cast<float>(features) + epsilon);
    }

    for (std::int32_t r = 0; r < features; ++r) {
        const float* xs = x_data + std::size_t(r) * stride + first;
        float* ys       = y_data + std::size_t(r) * stride + first;
        const __m256 vw = _mm256_set1_ps(weight[r]);
        const __m256 vb = _mm256_set1_ps(bias[r]);
        std::int32_t t  = 0;
        for (; t < vec_end; t += 8) {
            const __m256 c = _mm256_sub_ps(_mm256_loadu_ps(xs + t), _mm256_loadu_ps(mean.data() + t));
            const __m256 n = _mm256_mul_ps(_mm256_mul_ps(c, _mm256_loadu_ps(variance.data() + t)), vw);
            _mm256_storeu_ps(ys + t, _mm256_add_ps(n, vb));
        }
        for (; t < count; ++t) {
            ys[t] = (xs[t] - mean[t]) * variance[t] * weight[r] + bias[r];
        }
    }
}

void add_into(float* x, const float* delta, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(delta + i)));
    }
    for (; i < count; ++i) { x[i] += delta[i]; }
}

void gelu_tanh(float* x, std::size_t count) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0F);
    const __m256 half     = _mm256_set1_ps(0.5F);
    const __m256 one      = _mm256_set1_ps(1.0F);
    const __m256 kscale   = _mm256_set1_ps(0.7978845608028654F);
    const __m256 kcubic   = _mm256_set1_ps(0.044715F);
    const __m256 neg2     = _mm256_set1_ps(-2.0F);

    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 v         = _mm256_loadu_ps(x + i);
        const __m256 cube      = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        const __m256 inner     = _mm256_mul_ps(kscale, _mm256_add_ps(_mm256_mul_ps(cube, kcubic), v));
        const __m256 magnitude = _mm256_andnot_ps(sign_bit, inner);
        const __m256 t         = exp_nonpositive(_mm256_mul_ps(magnitude, neg2));
        __m256 th              = _mm256_div_ps(_mm256_sub_ps(one, t), _mm256_add_ps(one, t));
        th                     = _mm256_or_ps(th, _mm256_and_ps(sign_bit, inner));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, th)));
    }
    for (; i < count; ++i) {
        const float v     = x[i];
        const float inner = 0.7978845608028654F * (v + 0.044715F * v * v * v);
        const float mag   = std::abs(inner);
        const float t     = exp_nonpositive_scalar(-2.0F * mag);
        const float abs_t = (1.0F - t) / (1.0F + t);
        const float th    = inner < 0.0F ? -abs_t : abs_t;
        x[i]              = 0.5F * v * (1.0F + th);
    }
}

void rope_pair(float* first, float* second, const float* cosines, const float* sines,
               std::int32_t tokens) {
    if (tokens <= 0) { return; }
    std::int32_t t = 0;
    for (; t + 8 <= tokens; t += 8) {
        const __m256 a = _mm256_loadu_ps(first + t);
        const __m256 b = _mm256_loadu_ps(second + t);
        const __m256 c = _mm256_loadu_ps(cosines + t);
        const __m256 s = _mm256_loadu_ps(sines + t);
        _mm256_storeu_ps(first + t, _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s)));
        _mm256_storeu_ps(second + t, _mm256_add_ps(_mm256_mul_ps(b, c), _mm256_mul_ps(a, s)));
    }
    for (; t < tokens; ++t) {
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
    if (qn <= 0 || segment <= 0) { return; }
    constexpr std::int32_t kKeys = 256;
    const float* q_base          = qkv + std::size_t(row_base) * qkv_stride + q0;
    const float* k_base          = qkv + std::size_t(hidden + row_base) * qkv_stride + s0;
    const float* v_base          = qkv + std::size_t(2 * hidden + row_base) * qkv_stride + s0;

    for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
        const std::int32_t kn = std::min(kKeys, segment - k0);
        std::int32_t q        = 0;
        for (; q + 4 <= qn; q += 4) {
            std::int32_t key = 0;
            for (; key + 16 <= kn; key += 16) {
                __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const float* kp = k_base + std::size_t(d) * qkv_stride + k0 + key;
                    const float* qp = q_base + std::size_t(d) * qkv_stride + q;
                    const __m256 k0v = _mm256_loadu_ps(kp);
                    const __m256 k1v = _mm256_loadu_ps(kp + 8);
                    __m256 qc = _mm256_set1_ps(qp[0] * scale);
                    a00 = fmadd(qc, k0v, a00); a01 = fmadd(qc, k1v, a01);
                    qc  = _mm256_set1_ps(qp[1] * scale);
                    a10 = fmadd(qc, k0v, a10); a11 = fmadd(qc, k1v, a11);
                    qc  = _mm256_set1_ps(qp[2] * scale);
                    a20 = fmadd(qc, k0v, a20); a21 = fmadd(qc, k1v, a21);
                    qc  = _mm256_set1_ps(qp[3] * scale);
                    a30 = fmadd(qc, k0v, a30); a31 = fmadd(qc, k1v, a31);
                }
                _mm256_storeu_ps(scores + std::size_t(q) * segment + k0 + key, a00);
                _mm256_storeu_ps(scores + std::size_t(q) * segment + k0 + key + 8, a01);
                _mm256_storeu_ps(scores + std::size_t(q + 1) * segment + k0 + key, a10);
                _mm256_storeu_ps(scores + std::size_t(q + 1) * segment + k0 + key + 8, a11);
                _mm256_storeu_ps(scores + std::size_t(q + 2) * segment + k0 + key, a20);
                _mm256_storeu_ps(scores + std::size_t(q + 2) * segment + k0 + key + 8, a21);
                _mm256_storeu_ps(scores + std::size_t(q + 3) * segment + k0 + key, a30);
                _mm256_storeu_ps(scores + std::size_t(q + 3) * segment + k0 + key + 8, a31);
            }
            for (; key < kn; ++key) {
                float s0v = 0.0F, s1v = 0.0F, s2v = 0.0F, s3v = 0.0F;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const float kv  = k_base[std::size_t(d) * qkv_stride + k0 + key];
                    const float* qp = q_base + std::size_t(d) * qkv_stride + q;
                    s0v += qp[0] * scale * kv;
                    s1v += qp[1] * scale * kv;
                    s2v += qp[2] * scale * kv;
                    s3v += qp[3] * scale * kv;
                }
                scores[std::size_t(q) * segment + k0 + key]     = s0v;
                scores[std::size_t(q + 1) * segment + k0 + key] = s1v;
                scores[std::size_t(q + 2) * segment + k0 + key] = s2v;
                scores[std::size_t(q + 3) * segment + k0 + key] = s3v;
            }
        }
        for (; q < qn; ++q) {
            float* line      = scores + std::size_t(q) * segment + k0;
            std::int32_t key = 0;
            for (; key + 8 <= kn; key += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const __m256 qc = _mm256_set1_ps(q_base[std::size_t(d) * qkv_stride + q] * scale);
                    const __m256 kv = _mm256_loadu_ps(k_base + std::size_t(d) * qkv_stride + k0 + key);
                    acc             = fmadd(qc, kv, acc);
                }
                _mm256_storeu_ps(line + key, acc);
            }
            for (; key < kn; ++key) {
                float acc = 0.0F;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    acc += (q_base[std::size_t(d) * qkv_stride + q] * scale) *
                           k_base[std::size_t(d) * qkv_stride + k0 + key];
                }
                line[key] = acc;
            }
        }
    }

    for (std::int32_t q = 0; q < qn; ++q) {
        float* line      = scores + std::size_t(q) * segment;
        __m256 vmax      = _mm256_set1_ps(-INFINITY);
        std::int32_t key = 0;
        for (; key + 8 <= segment; key += 8) {
            vmax = max_nan_ps(vmax, _mm256_loadu_ps(line + key));
        }
        float m = hmax_f32_8(vmax);
        for (; key < segment; ++key) {
            if (!std::isnan(m)) {
                const float v = line[key];
                if (std::isnan(v) || v > m) { m = v; }
            }
        }

        const __m256 vm = _mm256_set1_ps(m);
        __m256 vsum     = _mm256_setzero_ps();
        key             = 0;
        for (; key + 8 <= segment; key += 8) {
            const __m256 e = exp_nonpositive(_mm256_sub_ps(_mm256_loadu_ps(line + key), vm));
            _mm256_storeu_ps(line + key, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        float sum = hsum_f32_8(vsum);
        for (; key < segment; ++key) {
            line[key] = std::exp(line[key] - m);
            sum += line[key];
        }
        const float inverse = 1.0F / sum;
        const __m256 vinv   = _mm256_set1_ps(inverse);
        key                 = 0;
        for (; key + 8 <= segment; key += 8) {
            _mm256_storeu_ps(line + key, _mm256_mul_ps(_mm256_loadu_ps(line + key), vinv));
        }
        for (; key < segment; ++key) { line[key] *= inverse; }
    }

    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        std::fill_n(out + std::size_t(row_base + d) * out_stride + q0, qn, 0.0F);
    }
    for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
        const std::int32_t kn = std::min(kKeys, segment - k0);
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const float* vd = v_base + std::size_t(d) * qkv_stride + k0;
            float* target   = out + std::size_t(row_base + d) * out_stride + q0;
            std::int32_t q  = 0;
            for (; q + 4 <= qn; q += 4) {
                const float* s0p = scores + std::size_t(q) * segment + k0;
                const float* s1p = s0p + segment;
                const float* s2p = s1p + segment;
                const float* s3p = s2p + segment;
                __m256 r0 = _mm256_setzero_ps(), r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps(), r3 = _mm256_setzero_ps();
                std::int32_t key = 0;
                for (; key + 8 <= kn; key += 8) {
                    const __m256 vv = _mm256_loadu_ps(vd + key);
                    r0              = fmadd(_mm256_loadu_ps(s0p + key), vv, r0);
                    r1              = fmadd(_mm256_loadu_ps(s1p + key), vv, r1);
                    r2              = fmadd(_mm256_loadu_ps(s2p + key), vv, r2);
                    r3              = fmadd(_mm256_loadu_ps(s3p + key), vv, r3);
                }
                float sum0 = hsum_f32_8(r0), sum1 = hsum_f32_8(r1);
                float sum2 = hsum_f32_8(r2), sum3 = hsum_f32_8(r3);
                for (; key < kn; ++key) {
                    const float vv = vd[key];
                    sum0 += s0p[key] * vv;
                    sum1 += s1p[key] * vv;
                    sum2 += s2p[key] * vv;
                    sum3 += s3p[key] * vv;
                }
                target[q]     += sum0;
                target[q + 1] += sum1;
                target[q + 2] += sum2;
                target[q + 3] += sum3;
            }
            for (; q < qn; ++q) {
                const float* sp  = scores + std::size_t(q) * segment + k0;
                __m256 r         = _mm256_setzero_ps();
                std::int32_t key = 0;
                for (; key + 8 <= kn; key += 8) {
                    r = fmadd(_mm256_loadu_ps(sp + key), _mm256_loadu_ps(vd + key), r);
                }
                float sum = hsum_f32_8(r);
                for (; key < kn; ++key) { sum += sp[key] * vd[key]; }
                target[q] += sum;
            }
        }
    }
}

} // namespace

#if defined(NINFER_QWEN_VISION_AVX2)
const Kernels& avx2_kernels() noexcept {
    static const Kernels kernels{pack_panel, gemm_packed_bias, layer_norm_cols,        add_into,
                                 gelu_tanh,  rope_pair,        segment_attention_block, "avx2"};
    return kernels;
}
#else
const Kernels& avx_kernels() noexcept {
    static const Kernels kernels{pack_panel, gemm_packed_bias, layer_norm_cols,        add_into,
                                 gelu_tanh,  rope_pair,        segment_attention_block, "avx"};
    return kernels;
}
#endif

} // namespace ninfer::models::qwen3_5::execution::cpu_vision
