#include "ops/moe_experts/cpu_projection.h"
#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::cpu_expert_detail {
namespace {

inline float half_f16c(const void* scales, std::size_t index) noexcept {
    std::uint16_t bits;
    std::memcpy(&bits, static_cast<const std::uint8_t*>(scales) + index * 2, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline int hsum_i32_4(__m128i x) noexcept {
    const __m128i hi64  = _mm_unpackhi_epi64(x, x);
    const __m128i sum64 = _mm_add_epi32(x, hi64);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

inline float hsum_f32_8(__m256 x) noexcept {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
}

inline float hmax_f32_8(__m256 x) noexcept {
    const __m128 s = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 h = _mm_max_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_max_ss(h, _mm_shuffle_ps(h, h, 1)));
}

inline __m256 bf16x8_to_ps(__m128i v) noexcept {
    const __m128i zero = _mm_setzero_si128();
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, v))),
        _mm_castsi128_ps(_mm_unpackhi_epi16(zero, v)), 1);
}

inline void i8x16_to_f32x8x2(__m128i w, __m256& out_lo, __m256& out_hi) noexcept {
    const __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(w));
    const __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(w, 4)));
    const __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(w, 8)));
    const __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(w, 12)));
    out_lo = _mm256_insertf128_ps(_mm256_castps128_ps256(f0), f1, 1);
    out_hi = _mm256_insertf128_ps(_mm256_castps128_ps256(f2), f3, 1);
}

inline int dot32_i8(__m128i w0, __m128i w1, const std::int8_t* in, __m128i ones16) noexcept {
    const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
    const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 16));
    const __m128i p0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
    const __m128i p1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));
    return hsum_i32_4(_mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16)));
}

inline int dot32_preabs_i8(__m128i aw0, __m128i aw1, __m128i w0, __m128i w1,
                           const std::int8_t* in, __m128i ones16) noexcept {
    const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
    const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 16));
    const __m128i p0 = _mm_maddubs_epi16(aw0, _mm_sign_epi8(x0, w0));
    const __m128i p1 = _mm_maddubs_epi16(aw1, _mm_sign_epi8(x1, w1));
    return hsum_i32_4(_mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16)));
}

inline float dot32_i8_f32(__m128i w0, __m128i w1, const float* in) noexcept {
    __m256 wf0, wf1, wf2, wf3;
    i8x16_to_f32x8x2(w0, wf0, wf1);
    i8x16_to_f32x8x2(w1, wf2, wf3);
    const __m256 s0 = _mm256_add_ps(
        _mm256_mul_ps(wf0, _mm256_loadu_ps(in)),
        _mm256_mul_ps(wf2, _mm256_loadu_ps(in + 16)));
    const __m256 s1 = _mm256_add_ps(
        _mm256_mul_ps(wf1, _mm256_loadu_ps(in + 8)),
        _mm256_mul_ps(wf3, _mm256_loadu_ps(in + 24)));
    return hsum_f32_8(_mm256_add_ps(s0, s1));
}

inline float dot32_f32_pre(const float* wf, const float* in) noexcept {
    const __m256 s0 = _mm256_add_ps(
        _mm256_mul_ps(_mm256_load_ps(wf), _mm256_loadu_ps(in)),
        _mm256_mul_ps(_mm256_load_ps(wf + 16), _mm256_loadu_ps(in + 16)));
    const __m256 s1 = _mm256_add_ps(
        _mm256_mul_ps(_mm256_load_ps(wf + 8), _mm256_loadu_ps(in + 8)),
        _mm256_mul_ps(_mm256_load_ps(wf + 24), _mm256_loadu_ps(in + 24)));
    return hsum_f32_8(_mm256_add_ps(s0, s1));
}

inline void unpack_q2_64(const std::uint8_t* low, __m128i& w0, __m128i& w1, __m128i& w2,
                         __m128i& w3) noexcept {
    const __m128i m3  = _mm_set1_epi8(0x03);
    const __m128i one = _mm_set1_epi8(1);
    const __m128i b   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
    const __m128i v0  = _mm_sub_epi8(_mm_and_si128(b, m3), one);
    const __m128i v1  = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 2), m3), one);
    const __m128i v2  = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), m3), one);
    const __m128i v3  = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 6), m3), one);
    const __m128i e01 = _mm_unpacklo_epi8(v0, v1);
    const __m128i o01 = _mm_unpackhi_epi8(v0, v1);
    const __m128i e23 = _mm_unpacklo_epi8(v2, v3);
    const __m128i o23 = _mm_unpackhi_epi8(v2, v3);
    w0 = _mm_unpacklo_epi16(e01, e23);
    w1 = _mm_unpackhi_epi16(e01, e23);
    w2 = _mm_unpacklo_epi16(o01, o23);
    w3 = _mm_unpackhi_epi16(o01, o23);
}

inline void unpack_u4_64(const std::uint8_t* low, __m128i& u0, __m128i& u1, __m128i& u2,
                         __m128i& u3) noexcept {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i b0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
    const __m128i b1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low + 16));
    const __m128i l0  = _mm_and_si128(b0, m0f);
    const __m128i h0  = _mm_and_si128(_mm_srli_epi16(b0, 4), m0f);
    const __m128i l1  = _mm_and_si128(b1, m0f);
    const __m128i h1  = _mm_and_si128(_mm_srli_epi16(b1, 4), m0f);
    u0 = _mm_unpacklo_epi8(l0, h0);
    u1 = _mm_unpackhi_epi8(l0, h0);
    u2 = _mm_unpacklo_epi8(l1, h1);
    u3 = _mm_unpackhi_epi8(l1, h1);
}

inline void unpack_q4_64(const std::uint8_t* low, __m128i& w0, __m128i& w1, __m128i& w2,
                         __m128i& w3) noexcept {
    const __m128i m0f  = _mm_set1_epi8(0x0F);
    const __m128i lut4 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    const __m128i b0   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
    const __m128i b1   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low + 16));
    const __m128i l0   = _mm_shuffle_epi8(lut4, _mm_and_si128(b0, m0f));
    const __m128i h0   = _mm_shuffle_epi8(lut4, _mm_and_si128(_mm_srli_epi16(b0, 4), m0f));
    const __m128i l1   = _mm_shuffle_epi8(lut4, _mm_and_si128(b1, m0f));
    const __m128i h1   = _mm_shuffle_epi8(lut4, _mm_and_si128(_mm_srli_epi16(b1, 4), m0f));
    w0 = _mm_unpacklo_epi8(l0, h0);
    w1 = _mm_unpackhi_epi8(l0, h0);
    w2 = _mm_unpacklo_epi8(l1, h1);
    w3 = _mm_unpackhi_epi8(l1, h1);
}

inline void unpack_q5_64(const std::uint8_t* low, const std::uint8_t* high, __m128i& w0,
                         __m128i& w1, __m128i& w2, __m128i& w3) noexcept {
    unpack_u4_64(low, w0, w1, w2, w3);
    const __m128i bit_mask = _mm_set1_epi64x(0x8040201008040201LL);
    const __m128i shuf_01  = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i shuf_23  = _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m128i shuf_45  = _mm_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5);
    const __m128i shuf_67  = _mm_setr_epi8(6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);
    const __m128i hb       = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high));
    const __m128i m10      = _mm_set1_epi8(0x10);
    const __m128i bias16   = _mm_set1_epi8(16);
    auto hi_bits = [&](__m128i sh) noexcept {
        const __m128i masked = _mm_and_si128(_mm_shuffle_epi8(hb, sh), bit_mask);
        return _mm_and_si128(_mm_cmpeq_epi8(masked, bit_mask), m10);
    };
    w0 = _mm_sub_epi8(_mm_xor_si128(_mm_or_si128(w0, hi_bits(shuf_01)), bias16), bias16);
    w1 = _mm_sub_epi8(_mm_xor_si128(_mm_or_si128(w1, hi_bits(shuf_23)), bias16), bias16);
    w2 = _mm_sub_epi8(_mm_xor_si128(_mm_or_si128(w2, hi_bits(shuf_45)), bias16), bias16);
    w3 = _mm_sub_epi8(_mm_xor_si128(_mm_or_si128(w3, hi_bits(shuf_67)), bias16), bias16);
}

inline void unpack_q6_64(const std::uint8_t* low, const std::uint8_t* high, __m128i& w0,
                         __m128i& w1, __m128i& w2, __m128i& w3) noexcept {
    unpack_u4_64(low, w0, w1, w2, w3);
    const __m128i hb     = _mm_loadu_si128(reinterpret_cast<const __m128i*>(high));
    const __m128i m03    = _mm_set1_epi8(0x03);
    const __m128i c0     = _mm_and_si128(hb, m03);
    const __m128i c1     = _mm_and_si128(_mm_srli_epi16(hb, 2), m03);
    const __m128i c2     = _mm_and_si128(_mm_srli_epi16(hb, 4), m03);
    const __m128i c3     = _mm_and_si128(_mm_srli_epi16(hb, 6), m03);
    const __m128i e01    = _mm_unpacklo_epi8(c0, c1);
    const __m128i o01    = _mm_unpackhi_epi8(c0, c1);
    const __m128i e23    = _mm_unpacklo_epi8(c2, c3);
    const __m128i o23    = _mm_unpackhi_epi8(c2, c3);
    const __m128i bias32 = _mm_set1_epi8(32);
    // Each unpacked byte in e01/e23/o01/o23 is in [0, 3], so shifting 16-bit lanes by 4
    // never carries into the upper byte.
    auto add_hi = [&](__m128i w, __m128i h) noexcept {
        return _mm_sub_epi8(_mm_xor_si128(_mm_or_si128(w, _mm_slli_epi16(h, 4)), bias32), bias32);
    };
    w0 = add_hi(w0, _mm_unpacklo_epi16(e01, e23));
    w1 = add_hi(w1, _mm_unpackhi_epi16(e01, e23));
    w2 = add_hi(w2, _mm_unpacklo_epi16(o01, o23));
    w3 = add_hi(w3, _mm_unpackhi_epi16(o01, o23));
}

template<int Bits>
inline void unpack_group(const std::uint8_t* low, const std::uint8_t* high, __m128i& w0,
                         __m128i& w1, __m128i& w2, __m128i& w3) noexcept {
    if constexpr (Bits == 2) {
        unpack_q2_64(low, w0, w1, w2, w3);
    } else if constexpr (Bits == 4) {
        unpack_q4_64(low, w0, w1, w2, w3);
    } else if constexpr (Bits == 5) {
        unpack_q5_64(low, high, w0, w1, w2, w3);
    } else if constexpr (Bits == 6) {
        unpack_q6_64(low, high, w0, w1, w2, w3);
    } else {
        w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
        w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low + 16));
    }
}

template<int Bits, bool Panel>
void quantized_avx(const Weight& weight, const Activations& x, bool a8, float* output) {
    constexpr int group_size = Bits == 8 ? 32 : 64;
    constexpr std::size_t low_stride  = Bits == 2 ? 16 : 32;
    constexpr std::size_t high_stride = Bits == 5 ? 8 : (Bits == 6 ? 16 : 0);
    constexpr std::size_t rec_step    = Panel ? kRowSplitPanelRows : 1;

    const int n      = weight.n;
    const int k      = weight.k;
    const int groups = k / group_size;
    const int blocks = k / 32;
    const auto* qdata = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* qhigh = static_cast<const std::uint8_t*>(weight.qhigh);
    const __m128i ones16 = _mm_set1_epi16(1);

    if (x.tokens == 1) {
        for (int row = 0; row < n; ++row) {
            if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) {
                throw CpuExpertCancelled();
            }
            const std::size_t rec_base = Panel
                ? (std::size_t(row / kRowSplitPanelRows) * groups) * kRowSplitPanelRows +
                      std::size_t(row % kRowSplitPanelRows)
                : std::size_t(row) * groups;
            const std::size_t scale_base = std::size_t(row) * groups;
            float acc = 0.0F;

            for (int g = 0; g < groups; ++g) {
                const std::size_t record = rec_base + std::size_t(g) * rec_step;
                const auto* low  = qdata + record * low_stride;
                const auto* high = (Bits == 5 || Bits == 6) ? (qhigh + record * high_stride) : nullptr;
                const float scale = half_f16c(weight.scales, scale_base + g);

                __m128i w0, w1, w2, w3;
                unpack_group<Bits>(low, high, w0, w1, w2, w3);

                if (a8) {
                    if constexpr (Bits == 8) {
                        const int d = dot32_i8(w0, w1, x.codes + g * 32, ones16);
                        acc += (float(d) * x.scales[g]) * scale;
                    } else {
                        const int b0 = g * 2;
                        const int d0 = dot32_i8(w0, w1, x.codes + g * 64, ones16);
                        const int d1 = dot32_i8(w2, w3, x.codes + g * 64 + 32, ones16);
                        acc += (float(d0) * x.scales[b0]) * scale;
                        acc += (float(d1) * x.scales[b0 + 1]) * scale;
                    }
                } else {
                    if constexpr (Bits == 8) {
                        acc += dot32_i8_f32(w0, w1, x.values + g * 32) * scale;
                    } else {
                        acc += dot32_i8_f32(w0, w1, x.values + g * 64) * scale;
                        acc += dot32_i8_f32(w2, w3, x.values + g * 64 + 32) * scale;
                    }
                }
            }
            output[row] = acc;
        }
        return;
    }

    constexpr int kTokChunk = 16;
    const std::int8_t* xc_ptrs[kTokChunk];
    const float* xs_ptrs[kTokChunk];
    const float* xf_ptrs[kTokChunk];

    for (int t0 = 0; t0 < x.tokens; t0 += kTokChunk) {
        const int tc = std::min(kTokChunk, x.tokens - t0);
        for (int t = 0; t < tc; ++t) {
            const int tok = t0 + t;
            if (a8) {
                xc_ptrs[t] = x.codes + std::size_t(tok) * k;
                xs_ptrs[t] = x.scales + std::size_t(tok) * blocks;
            } else {
                xf_ptrs[t] = x.values + std::size_t(tok) * k;
            }
        }
        for (int row = 0; row < n; ++row) {
            if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) {
                throw CpuExpertCancelled();
            }
            const std::size_t rec_base = Panel
                ? (std::size_t(row / kRowSplitPanelRows) * groups) * kRowSplitPanelRows +
                      std::size_t(row % kRowSplitPanelRows)
                : std::size_t(row) * groups;
            const std::size_t scale_base = std::size_t(row) * groups;
            float acc[kTokChunk] = {};

            for (int g = 0; g < groups; ++g) {
                const std::size_t record = rec_base + std::size_t(g) * rec_step;
                const auto* low  = qdata + record * low_stride;
                const auto* high = (Bits == 5 || Bits == 6) ? (qhigh + record * high_stride) : nullptr;
                const float scale = half_f16c(weight.scales, scale_base + g);

                __m128i w0, w1, w2, w3;
                unpack_group<Bits>(low, high, w0, w1, w2, w3);

                if (a8) {
                    const __m128i aw0 = _mm_abs_epi8(w0);
                    const __m128i aw1 = _mm_abs_epi8(w1);
                    if constexpr (Bits == 8) {
                        const int col = g * 32;
                        for (int t = 0; t < tc; ++t) {
                            const int d = dot32_preabs_i8(aw0, aw1, w0, w1, xc_ptrs[t] + col, ones16);
                            acc[t] += (float(d) * xs_ptrs[t][g]) * scale;
                        }
                    } else {
                        const __m128i aw2 = _mm_abs_epi8(w2);
                        const __m128i aw3 = _mm_abs_epi8(w3);
                        const int col = g * 64;
                        const int b0  = g * 2;
                        for (int t = 0; t < tc; ++t) {
                            const int d0 = dot32_preabs_i8(aw0, aw1, w0, w1, xc_ptrs[t] + col, ones16);
                            const int d1 = dot32_preabs_i8(aw2, aw3, w2, w3, xc_ptrs[t] + col + 32, ones16);
                            acc[t] += (float(d0) * xs_ptrs[t][b0]) * scale;
                            acc[t] += (float(d1) * xs_ptrs[t][b0 + 1]) * scale;
                        }
                    }
                } else {
                    alignas(32) float wf[group_size];
                    __m256 f0, f1;
                    i8x16_to_f32x8x2(w0, f0, f1);
                    _mm256_store_ps(wf, f0);
                    _mm256_store_ps(wf + 8, f1);
                    i8x16_to_f32x8x2(w1, f0, f1);
                    _mm256_store_ps(wf + 16, f0);
                    _mm256_store_ps(wf + 24, f1);
                    if constexpr (Bits != 8) {
                        i8x16_to_f32x8x2(w2, f0, f1);
                        _mm256_store_ps(wf + 32, f0);
                        _mm256_store_ps(wf + 40, f1);
                        i8x16_to_f32x8x2(w3, f0, f1);
                        _mm256_store_ps(wf + 48, f0);
                        _mm256_store_ps(wf + 56, f1);
                    }
                    const int col = g * group_size;
                    for (int t = 0; t < tc; ++t) {
                        acc[t] += dot32_f32_pre(wf, xf_ptrs[t] + col) * scale;
                        if constexpr (Bits != 8) {
                            acc[t] += dot32_f32_pre(wf + 32, xf_ptrs[t] + col + 32) * scale;
                        }
                    }
                }
            }
            for (int t = 0; t < tc; ++t) {
                output[std::size_t(t0 + t) * n + row] = acc[t];
            }
        }
    }
}

void project_bf16_avx(const Weight& weight, const Activations& x, float* output) {
    const int n = weight.n;
    const int k = weight.k;
    const auto* wbytes = static_cast<const std::uint8_t*>(weight.qdata);

    if (x.tokens == 1) {
        const float* xv = x.values;
        for (int row = 0; row < n; ++row) {
            if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) {
                throw CpuExpertCancelled();
            }
            const auto* row_ptr = wbytes + std::size_t(row) * k * 2;
            __m256 sum0 = _mm256_setzero_ps();
            int col = 0;
            for (; col + 8 <= k; col += 8) {
                const __m256 w0 = bf16x8_to_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + col * 2)));
                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(w0, _mm256_loadu_ps(xv + col)));
            }
            float acc = hsum_f32_8(sum0);
            for (; col < k; ++col) {
                std::uint16_t raw;
                std::memcpy(&raw, row_ptr + col * 2, 2);
                acc += std::bit_cast<float>(std::uint32_t(raw) << 16) * xv[col];
            }
            output[row] = acc;
        }
        return;
    }

    for (int t0 = 0; t0 < x.tokens; t0 += 4) {
        const int tc = std::min(4, x.tokens - t0);
        const float* x0 = x.values + std::size_t(t0) * k;
        const float* x1 = tc > 1 ? x0 + k : x0;
        const float* x2 = tc > 2 ? x0 + 2 * k : x0;
        const float* x3 = tc > 3 ? x0 + 3 * k : x0;

        for (int row = 0; row < n; ++row) {
            if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) {
                throw CpuExpertCancelled();
            }
            const auto* row_ptr = wbytes + std::size_t(row) * k * 2;
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            int col = 0;
            for (; col + 8 <= k; col += 8) {
                const __m256 w = bf16x8_to_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + col * 2)));
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(w, _mm256_loadu_ps(x0 + col)));
                if (tc > 1) { s1 = _mm256_add_ps(s1, _mm256_mul_ps(w, _mm256_loadu_ps(x1 + col))); }
                if (tc > 2) { s2 = _mm256_add_ps(s2, _mm256_mul_ps(w, _mm256_loadu_ps(x2 + col))); }
                if (tc > 3) { s3 = _mm256_add_ps(s3, _mm256_mul_ps(w, _mm256_loadu_ps(x3 + col))); }
            }
            float a0 = hsum_f32_8(s0);
            float a1 = tc > 1 ? hsum_f32_8(s1) : 0.0F;
            float a2 = tc > 2 ? hsum_f32_8(s2) : 0.0F;
            float a3 = tc > 3 ? hsum_f32_8(s3) : 0.0F;
            for (; col < k; ++col) {
                std::uint16_t raw;
                std::memcpy(&raw, row_ptr + col * 2, 2);
                const float w = std::bit_cast<float>(std::uint32_t(raw) << 16);
                a0 += w * x0[col];
                if (tc > 1) { a1 += w * x1[col]; }
                if (tc > 2) { a2 += w * x2[col]; }
                if (tc > 3) { a3 += w * x3[col]; }
            }
            output[std::size_t(t0) * n + row] = a0;
            if (tc > 1) { output[std::size_t(t0 + 1) * n + row] = a1; }
            if (tc > 2) { output[std::size_t(t0 + 2) * n + row] = a2; }
            if (tc > 3) { output[std::size_t(t0 + 3) * n + row] = a3; }
        }
    }
}

inline __m128i pack_i32_to_i8(__m256 v_lo, __m256 v_hi) noexcept {
    const __m256i q0 = _mm256_cvtps_epi32(v_lo);
    const __m256i q1 = _mm256_cvtps_epi32(v_hi);
    const __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extractf128_si256(q0, 1));
    const __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extractf128_si256(q1, 1));
    return _mm_packs_epi16(p0, p1);
}

inline void quantize32_avx(__m256 v0, __m256 v1, __m256 v2, __m256 v3,
                           std::int8_t* codes, float* scale_out) noexcept {
    const __m256 absmask = _mm256_set1_ps(std::bit_cast<float>(0x7FFFFFFFU));
    const __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, absmask), _mm256_and_ps(v1, absmask));
    const __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, absmask), _mm256_and_ps(v3, absmask));
    const float maximum = hmax_f32_8(_mm256_max_ps(m01, m23));
    const float scale = maximum / 127.0F;
    *scale_out = scale;
    if (scale == 0.0F) {
        const __m128i z = _mm_setzero_si128();
        _mm_storeu_si128(reinterpret_cast<__m128i*>(codes), z);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + 16), z);
        return;
    }
    const __m256 vs  = _mm256_set1_ps(scale);
    const __m256 lo  = _mm256_set1_ps(-127.0F);
    const __m256 hi  = _mm256_set1_ps(127.0F);
    auto clamp_div = [&](__m256 v) noexcept {
        return _mm256_min_ps(hi, _mm256_max_ps(lo, _mm256_div_ps(v, vs)));
    };
    _mm_storeu_si128(reinterpret_cast<__m128i*>(codes),      pack_i32_to_i8(clamp_div(v0), clamp_div(v1)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + 16), pack_i32_to_i8(clamp_div(v2), clamp_div(v3)));
}

inline __m256 exp256_ps(__m256 y) noexcept {
    const __m256 lo = _mm256_set1_ps(-87.0F);
    const __m256 hi = _mm256_set1_ps(88.0F);
    const __m256 z  = _mm256_max_ps(_mm256_min_ps(y, hi), lo);

    const __m256 log2e  = _mm256_set1_ps(1.44269504F);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125F);
    const __m256 ln2_lo = _mm256_set1_ps(1.42860677e-6F);
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(z, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(z, _mm256_mul_ps(n, ln2_hi));
    r        = _mm256_sub_ps(r, _mm256_mul_ps(n, ln2_lo));

    __m256 p = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.9875691500e-4F), r),
                             _mm256_set1_ps(1.3981999507e-3F));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(8.3334519073e-3F));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(4.1665795894e-2F));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.6666665459e-1F));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.0000001201e-1F));
    p = _mm256_add_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0F));

    const __m256i ni      = _mm256_cvtps_epi32(n);
    const __m128i bias127 = _mm_set1_epi32(127);
    const __m128i pow_lo  = _mm_slli_epi32(_mm_add_epi32(_mm256_castsi256_si128(ni), bias127), 23);
    const __m128i pow_hi  = _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256(ni, 1), bias127), 23);
    const __m256 pow2 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(pow_lo)), _mm_castsi128_ps(pow_hi), 1);

    __m256 res = _mm256_mul_ps(p, pow2);
    const __m256 is_udf = _mm256_cmp_ps(y, lo, _CMP_LT_OQ);
    const __m256 is_ovf = _mm256_cmp_ps(y, hi, _CMP_GE_OQ);
    const __m256 is_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
    res = _mm256_blendv_ps(res, _mm256_setzero_ps(), is_udf);
    res = _mm256_blendv_ps(res, _mm256_set1_ps(std::bit_cast<float>(0x7F800000U)), is_ovf);
    res = _mm256_blendv_ps(res, y, is_nan);
    return res;
}

template<int Bits>
void dispatch_layout(const Weight& w, const Activations& x, bool a8, float* y) {
    if (w.layout == QuantLayout::RowSplitPanel) {
        quantized_avx<Bits, true>(w, x, a8, y);
    } else {
        quantized_avx<Bits, false>(w, x, a8, y);
    }
}

} // namespace

void project_avx(const Weight& w, const Activations& x, bool a8, float* y) {
    switch (w.qtype) {
    case QType::Q2_G64_FP16: return dispatch_layout<2>(w, x, a8, y);
    case QType::Q4_G64_FP16: return dispatch_layout<4>(w, x, a8, y);
    case QType::Q5_G64_FP16: return dispatch_layout<5>(w, x, a8, y);
    case QType::Q6_G64_FP16: return dispatch_layout<6>(w, x, a8, y);
    case QType::Q8_G32_FP16: return dispatch_layout<8>(w, x, a8, y);
    default: return project_bf16_avx(w, x, y);
    }
}

void prepare_input_avx(const std::uint16_t* x, std::size_t count, float* values,
                       std::int8_t* codes, float* scales, bool a8) {
    std::size_t start = 0;
    if (!a8) {
        for (; start + 16 <= count; start += 16) {
            const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start));
            const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start + 8));
            _mm256_storeu_ps(values + start,     bf16x8_to_ps(b0));
            _mm256_storeu_ps(values + start + 8, bf16x8_to_ps(b1));
        }
        for (; start < count; ++start) {
            values[start] = std::bit_cast<float>(std::uint32_t(x[start]) << 16);
        }
        return;
    }

    const __m128i inf_mask = _mm_set1_epi16(0x7F80);
    for (; start + 32 <= count; start += 32) {
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start + 8));
        const __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start + 16));
        const __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + start + 24));
        const __m128i bad01 = _mm_or_si128(
            _mm_cmpeq_epi16(_mm_and_si128(b0, inf_mask), inf_mask),
            _mm_cmpeq_epi16(_mm_and_si128(b1, inf_mask), inf_mask));
        const __m128i bad23 = _mm_or_si128(
            _mm_cmpeq_epi16(_mm_and_si128(b2, inf_mask), inf_mask),
            _mm_cmpeq_epi16(_mm_and_si128(b3, inf_mask), inf_mask));
        const __m128i bad = _mm_or_si128(bad01, bad23);
        if (!_mm_testz_si128(bad, bad)) {
            throw std::runtime_error("CPU expert: nonfinite activation");
        }
        const __m256 v0 = bf16x8_to_ps(b0);
        const __m256 v1 = bf16x8_to_ps(b1);
        const __m256 v2 = bf16x8_to_ps(b2);
        const __m256 v3 = bf16x8_to_ps(b3);
        _mm256_storeu_ps(values + start,      v0);
        _mm256_storeu_ps(values + start + 8,  v1);
        _mm256_storeu_ps(values + start + 16, v2);
        _mm256_storeu_ps(values + start + 24, v3);
        quantize32_avx(v0, v1, v2, v3, codes + start, scales + start / 32);
    }
    for (; start < count; ++start) {
        values[start] = std::bit_cast<float>(std::uint32_t(x[start]) << 16);
    }
}

void swiglu_quantize_avx(const float* gate, float* middle, std::size_t count,
                         std::int8_t* codes, float* scales, bool a8) {
    const __m256 one  = _mm256_set1_ps(1.0F);
    const __m256 zero = _mm256_setzero_ps();
    std::size_t start = 0;

    auto silu_mul = [&](const float* gp, const float* mp) noexcept {
        const __m256 g = _mm256_loadu_ps(gp);
        const __m256 u = _mm256_loadu_ps(mp);
        const __m256 neg_g = _mm256_sub_ps(zero, g);
        const __m256 sig_g = _mm256_div_ps(g, _mm256_add_ps(one, exp256_ps(neg_g)));
        return _mm256_mul_ps(u, sig_g);
    };

    for (; start + 32 <= count; start += 32) {
        const __m256 m0 = silu_mul(gate + start,      middle + start);
        const __m256 m1 = silu_mul(gate + start + 8,  middle + start + 8);
        const __m256 m2 = silu_mul(gate + start + 16, middle + start + 16);
        const __m256 m3 = silu_mul(gate + start + 24, middle + start + 24);
        _mm256_storeu_ps(middle + start,      m0);
        _mm256_storeu_ps(middle + start + 8,  m1);
        _mm256_storeu_ps(middle + start + 16, m2);
        _mm256_storeu_ps(middle + start + 24, m3);
        if (a8) {
            const __m256 d0 = _mm256_sub_ps(m0, m0);
            const __m256 d1 = _mm256_sub_ps(m1, m1);
            const __m256 d2 = _mm256_sub_ps(m2, m2);
            const __m256 d3 = _mm256_sub_ps(m3, m3);
            const __m256 bad01 = _mm256_or_ps(_mm256_cmp_ps(d0, d0, _CMP_UNORD_Q),
                                              _mm256_cmp_ps(d1, d1, _CMP_UNORD_Q));
            const __m256 bad23 = _mm256_or_ps(_mm256_cmp_ps(d2, d2, _CMP_UNORD_Q),
                                              _mm256_cmp_ps(d3, d3, _CMP_UNORD_Q));
            if (_mm256_movemask_ps(_mm256_or_ps(bad01, bad23)) != 0) {
                throw std::runtime_error("CPU expert: nonfinite activation");
            }
            quantize32_avx(m0, m1, m2, m3, codes + start, scales + start / 32);
        }
    }
    for (; start < count; ++start) {
        const float g = gate[start];
        middle[start] *= g / (1.0F + std::exp(-g));
    }
}

} // namespace ninfer::ops::cpu_expert_detail
