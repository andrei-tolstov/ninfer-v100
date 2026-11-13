#include "ops/moe_experts/gguf_expert_cpu.h"

#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_cpu_tables.h"

#include <immintrin.h>

#include <cstring>

// The scalar products' arithmetic (gguf_expert_cpu.cpp) eight slices at a time using AVX 1.0,
// F16C, SSSE3 and SSE4.1 without AVX2 or FMA instructions. Group g holds slices 8 g .. 8 g + 7,
// which are GPU lanes 8 (g mod 4) .. 8 (g mod 4) + 7: one vector of each token's four holds their
// partial sums, and groups g and g + 4 add to it in slice order.

namespace ninfer::ops::gguf_cpu_detail {
namespace {

inline float half_at(const std::uint8_t* p) {
    std::uint16_t bits;
    std::memcpy(&bits, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline __m128i dot_u8s8_128(__m128i codes, __m128i a) {
    return _mm_madd_epi16(_mm_maddubs_epi16(codes, a), _mm_set1_epi16(1));
}

inline std::uint32_t load32(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

inline void sign_bytes(std::uint32_t bits, __m128i& out_lo, __m128i& out_hi) {
    const __m128i spread_lo = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i spread_hi = _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m128i select    = _mm_set1_epi64x(0x8040201008040201LL);
    const __m128i b_lo      = _mm_shuffle_epi8(_mm_set1_epi32(static_cast<int>(bits)), spread_lo);
    const __m128i b_hi      = _mm_shuffle_epi8(_mm_set1_epi32(static_cast<int>(bits)), spread_hi);
    const __m128i set_lo    = _mm_and_si128(b_lo, select);
    const __m128i set_hi    = _mm_and_si128(b_hi, select);
    out_lo = _mm_or_si128(_mm_cmpeq_epi8(set_lo, select), _mm_set1_epi8(1));
    out_hi = _mm_or_si128(_mm_cmpeq_epi8(set_hi, select), _mm_set1_epi8(1));
}

inline void iq4_values(const std::uint8_t* q, __m128i& out_lo, __m128i& out_hi) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    const __m128i nib   = _mm_set1_epi8(15);
    out_lo = _mm_shuffle_epi8(table, _mm_and_si128(bytes, nib));
    out_hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), nib));
}

struct Group {
    __m128i magnitude[8][2], sign[8][2];
    alignas(32) float factor[2][8];
};

inline void sum_hadd8(const __m128i* u, __m128i& out0123, __m128i& out4567) {
    const __m128i h01 = _mm_hadd_epi32(u[0], u[1]);
    const __m128i h23 = _mm_hadd_epi32(u[2], u[3]);
    out0123 = _mm_hadd_epi32(h01, h23);
    const __m128i h45 = _mm_hadd_epi32(u[4], u[5]);
    const __m128i h67 = _mm_hadd_epi32(u[6], u[7]);
    out4567 = _mm_hadd_epi32(h45, h67);
}

inline float butterfly8(const __m256* lanes) {
    const __m256 c = _mm256_add_ps(_mm256_add_ps(lanes[0], lanes[2]), _mm256_add_ps(lanes[1], lanes[3]));
    const __m128 d = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
    const __m128 e = _mm_add_ps(d, _mm_movehl_ps(d, d));
    return _mm_cvtss_f32(_mm_add_ss(e, _mm_shuffle_ps(e, e, 1)));
}

inline __m256 live_floats(const float* from, int live) {
    if (live == 8) { return _mm256_loadu_ps(from); }
    alignas(32) float to[8];
    for (int i = 0; i < 8; ++i) { to[i] = i < live ? from[i] : 0.0F; }
    return _mm256_load_ps(to);
}

// Emulated single-rounding fmaf using AVX 1.0 double-precision TwoSum + Boldo-Melquiond
// round-to-odd (IEEE Trans. Computers 57(4), 2008). Every float product is exact in double
// (48 <= 53 significand bits); rounding the TwoSum result to odd prevents double rounding when
// narrowing back to float. Compiled with -ffp-contract=off.
inline __m128 fma4(__m128 af, __m128 bf, __m128 cf) {
    const __m256d p    = _mm256_mul_pd(_mm256_cvtps_pd(af), _mm256_cvtps_pd(bf));
    const __m256d cd   = _mm256_cvtps_pd(cf);
    const __m256d s    = _mm256_add_pd(p, cd);
    const __m256d bb   = _mm256_sub_pd(s, p);
    const __m256d e    = _mm256_add_pd(_mm256_sub_pd(p, _mm256_sub_pd(s, bb)), _mm256_sub_pd(cd, bb));
    const __m256d zero = _mm256_setzero_pd();
    // Adjust only finite non-zero s whose TwoSum residual e is non-zero (ordered comparisons).
    const __m256d valid = _mm256_and_pd(
        _mm256_and_pd(_mm256_cmp_pd(e, zero, _CMP_NEQ_OQ), _mm256_cmp_pd(s, zero, _CMP_NEQ_OQ)),
        _mm256_cmp_pd(_mm256_sub_pd(s, s), zero, _CMP_EQ_OQ));
    const __m256d opp = _mm256_xor_pd(s, e);

    auto adjust128 = [](__m128d sd, __m128d vd, __m128d opd) {
        const __m128i bs       = _mm_castpd_si128(sd);
        const __m128i odd_hi   = _mm_srai_epi32(_mm_slli_epi64(bs, 63), 31);
        const __m128i is_odd   = _mm_shuffle_epi32(odd_hi, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i need     = _mm_andnot_si128(is_odd, _mm_castpd_si128(vd));
        const __m128i xor_hi   = _mm_srai_epi32(_mm_castpd_si128(opd), 31);
        const __m128i opp_sign = _mm_shuffle_epi32(xor_hi, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i step     = _mm_or_si128(opp_sign, _mm_set1_epi64x(1));
        return _mm_castsi128_pd(_mm_add_epi64(bs, _mm_and_si128(need, step)));
    };
    const __m128d lo = adjust128(_mm256_castpd256_pd128(s), _mm256_castpd256_pd128(valid),
                                 _mm256_castpd256_pd128(opp));
    const __m128d hi = adjust128(_mm256_extractf128_pd(s, 1), _mm256_extractf128_pd(valid, 1),
                                 _mm256_extractf128_pd(opp, 1));
    return _mm256_cvtpd_ps(_mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1));
}

inline __m256 fma8(__m256 a, __m256 b, __m256 c) {
    const __m128 lo = fma4(_mm256_castps256_ps128(a), _mm256_castps256_ps128(b),
                           _mm256_castps256_ps128(c));
    const __m128 hi = fma4(_mm256_extractf128_ps(a, 1), _mm256_extractf128_ps(b, 1),
                           _mm256_extractf128_ps(c, 1));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

constexpr int kMaxTokens = 16;

template <bool kHalves, class Decode, class Dot>
void rows_in_lane_order(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride, const Decode& decode,
                        const Dot& dot) {
    const int tokens = x.tokens();
    const int slices = w.k / 32;
    const int groups = (slices + 7) / 8;
    __m256 acc[kMaxTokens][4];
    Group group;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        for (int t = 0; t < tokens; ++t) {
            for (__m256& v : acc[t]) { v = _mm256_setzero_ps(); }
        }
        for (int g = 0; g < groups; ++g) {
            const int live = slices - 8 * g < 8 ? slices - 8 * g : 8;
            decode(r, g, live, group);
            const __m256 factor0 = _mm256_load_ps(group.factor[0]);
            for (int t = 0; t < tokens; ++t) {
                const __m256 a = live_floats(x.scales(t) + 8 * g, live);
                __m256 product;
                if constexpr (kHalves) {
                    const __m256 inner = dot(t, g, live, group);
                    product = _mm256_mul_ps(a, inner);
                } else {
                    product = _mm256_mul_ps(_mm256_mul_ps(factor0, a),
                                            dot(t, g, live, group));
                }
                acc[t][g & 3] = _mm256_add_ps(acc[t][g & 3], product);
            }
        }
        for (int t = 0; t < tokens; ++t) { y[std::size_t(t) * y_stride + row] = butterfly8(acc[t]); }
    }
}

template <QType F>
inline void decode_iq(const std::uint8_t* block, Group& out) {
    const float d = half_at(block);
    for (int sub = 0; sub < 8; ++sub) {
        if constexpr (F == QType::GGUF_IQ2_S) {
            const std::uint8_t* qs = block + 2 + 4 * sub;
            const unsigned qh      = block[66 + sub];
            const unsigned scale   = block[74 + sub];
            out.magnitude[sub][0]  = _mm_set_epi64x(
                static_cast<long long>(iq2s_grid[qs[1] | ((qh << 6) & 0x300U)]),
                static_cast<long long>(iq2s_grid[qs[0] | ((qh << 8) & 0x300U)]));
            out.magnitude[sub][1]  = _mm_set_epi64x(
                static_cast<long long>(iq2s_grid[qs[3] | ((qh << 2) & 0x300U)]),
                static_cast<long long>(iq2s_grid[qs[2] | ((qh << 4) & 0x300U)]));
            sign_bytes(load32(block + 34 + 4 * sub), out.sign[sub][0], out.sign[sub][1]);
            out.factor[0][sub] = d * float(2 * (scale & 15U) + 1) * 0.125F;
            out.factor[1][sub] = d * float(2 * (scale >> 4) + 1) * 0.125F;
        } else if constexpr (F == QType::GGUF_IQ3_XXS) {
            const std::uint8_t* qs  = block + 2 + 8 * sub;
            const std::uint32_t aux = load32(block + 66 + 4 * sub);
            out.magnitude[sub][0]   = _mm_set_epi32(
                static_cast<int>(iq3xxs_grid[qs[3]]), static_cast<int>(iq3xxs_grid[qs[2]]),
                static_cast<int>(iq3xxs_grid[qs[1]]), static_cast<int>(iq3xxs_grid[qs[0]]));
            out.magnitude[sub][1]   = _mm_set_epi32(
                static_cast<int>(iq3xxs_grid[qs[7]]), static_cast<int>(iq3xxs_grid[qs[6]]),
                static_cast<int>(iq3xxs_grid[qs[5]]), static_cast<int>(iq3xxs_grid[qs[4]]));
            const std::uint32_t bits = std::uint32_t(ksigns_iq2xs[aux & 127U]) |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 7) & 127U]) << 8 |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 14) & 127U]) << 16 |
                                       std::uint32_t(ksigns_iq2xs[(aux >> 21) & 127U]) << 24;
            sign_bytes(bits, out.sign[sub][0], out.sign[sub][1]);
            out.factor[0][sub] = d * float(2 * (aux >> 28) + 1) * 0.25F;
            out.factor[1][sub] = 0.0F;
        } else if constexpr (F == QType::GGUF_IQ3_S) {
            const std::uint8_t* qs = block + 2 + 8 * sub;
            const unsigned qh      = block[66 + sub];
            const unsigned scale   = (block[106 + sub / 2] >> (4 * (sub & 1))) & 15U;
            out.magnitude[sub][0]  = _mm_set_epi32(
                static_cast<int>(iq3s_grid[qs[3] | ((qh << 5) & 256U)]),
                static_cast<int>(iq3s_grid[qs[2] | ((qh << 6) & 256U)]),
                static_cast<int>(iq3s_grid[qs[1] | ((qh << 7) & 256U)]),
                static_cast<int>(iq3s_grid[qs[0] | ((qh << 8) & 256U)]));
            out.magnitude[sub][1]  = _mm_set_epi32(
                static_cast<int>(iq3s_grid[qs[7] | ((qh << 1) & 256U)]),
                static_cast<int>(iq3s_grid[qs[6] | ((qh << 2) & 256U)]),
                static_cast<int>(iq3s_grid[qs[5] | ((qh << 3) & 256U)]),
                static_cast<int>(iq3s_grid[qs[4] | ((qh << 4) & 256U)]));
            sign_bytes(load32(block + 74 + 4 * sub), out.sign[sub][0], out.sign[sub][1]);
            out.factor[0][sub] = d * float(2 * scale + 1);
            out.factor[1][sub] = 0.0F;
        } else {
            static_assert(F == QType::GGUF_IQ4_XS);
            std::uint16_t high;
            std::memcpy(&high, block + 2, 2);
            const unsigned low  = (block[4 + sub / 2] >> (4 * (sub & 1))) & 15U;
            __m128i value_lo, value_hi;
            iq4_values(block + 8 + 16 * sub, value_lo, value_hi);
            out.magnitude[sub][0] = _mm_sign_epi8(value_lo, value_lo);
            out.magnitude[sub][1] = _mm_sign_epi8(value_hi, value_hi);
            out.sign[sub][0]      = value_lo;
            out.sign[sub][1]      = value_hi;
            out.factor[0][sub]    = d * float(int(low | (((unsigned(high) >> (2 * sub)) & 3U) << 4)) - 32);
            out.factor[1][sub]    = 0.0F;
        }
    }
}

template <QType F>
void iq_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    constexpr int bytes   = gguf_block_shape(F).bytes;
    constexpr bool halves = F == QType::GGUF_IQ2_S;
    const auto decode     = [](const std::uint8_t* row, int g, int, Group& group) {
        decode_iq<F>(row + std::size_t(g) * bytes, group);
    };
    const auto dot = [&x](int t, int g, int, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m128i u_lo[8], u_hi[8];
        for (int i = 0; i < 8; ++i) {
            const __m128i v_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 32 * i));
            const __m128i v_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 32 * i + 16));
            u_lo[i] = dot_u8s8_128(group.magnitude[i][0], _mm_sign_epi8(v_lo, group.sign[i][0]));
            u_hi[i] = dot_u8s8_128(group.magnitude[i][1], _mm_sign_epi8(v_hi, group.sign[i][1]));
        }
        __m128i lo0123, lo4567, hi0123, hi4567;
        sum_hadd8(u_lo, lo0123, lo4567);
        sum_hadd8(u_hi, hi0123, hi4567);
        if constexpr (halves) {
            const __m256 d0 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_cvtepi32_ps(lo0123)), _mm_cvtepi32_ps(lo4567), 1);
            const __m256 d1 = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_cvtepi32_ps(hi0123)), _mm_cvtepi32_ps(hi4567), 1);
            const __m256 f0 = _mm256_load_ps(group.factor[0]);
            const __m256 f1 = _mm256_load_ps(group.factor[1]);
            return fma8(f0, d0, _mm256_mul_ps(f1, d1));
        } else {
            const __m128i tot_lo = _mm_add_epi32(lo0123, hi0123);
            const __m128i tot_hi = _mm_add_epi32(lo4567, hi4567);
            return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(tot_lo)),
                                        _mm_cvtepi32_ps(tot_hi), 1);
        }
    };
    rows_in_lane_order<halves>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

void q2_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    const auto decode = [](const std::uint8_t* row, int g, int live, Group& group) {
        const __m128i three = _mm_set1_epi8(3);
        for (int i = 0; i < 8; ++i) {
            group.factor[0][i] = i < live ? half_at(row + std::size_t(4 * g + i / 2) * 18) : 0.0F;
        }
        for (int b = 0; 2 * b < live; ++b) {
            const __m128i q =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + std::size_t(4 * g + b) * 18 + 2));
            group.magnitude[2 * b][0]     = _mm_and_si128(q, three);
            group.magnitude[2 * b][1]     = _mm_and_si128(_mm_srli_epi16(q, 2), three);
            group.magnitude[2 * b + 1][0] = _mm_and_si128(_mm_srli_epi16(q, 4), three);
            group.magnitude[2 * b + 1][1] = _mm_and_si128(_mm_srli_epi16(q, 6), three);
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        alignas(16) std::int32_t dots_arr[8];
        const std::int32_t* sums = x.sums(t) + 8 * g;
        for (int b = 0; b < 4; ++b) {
            if (2 * b >= live) {
                dots_arr[2 * b]     = 0;
                dots_arr[2 * b + 1] = 0;
                continue;
            }
            const std::int8_t* a = x.codes(t) + (4 * g + b) * 64;
            const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 16));
            const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 32));
            const __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 48));

            const __m128i p0 = dot_u8s8_128(group.magnitude[2 * b][0], a0);
            const __m128i p1 = dot_u8s8_128(group.magnitude[2 * b][1], a1);
            const __m128i p2 = dot_u8s8_128(group.magnitude[2 * b + 1][0], a2);
            const __m128i p3 = dot_u8s8_128(group.magnitude[2 * b + 1][1], a3);

            const __m128i lo = _mm_add_epi32(p0, p2);
            const __m128i hi = _mm_add_epi32(p1, p3);

            alignas(16) std::int32_t d_lo[4], d_hi[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(d_lo), lo);
            _mm_store_si128(reinterpret_cast<__m128i*>(d_hi), hi);

            const std::int32_t s0 = (d_lo[0] + d_lo[1]) + (d_hi[0] + d_hi[1]);
            const std::int32_t s1 = (d_lo[2] + d_lo[3]) + (d_hi[2] + d_hi[3]);

            dots_arr[2 * b]     = s0 - (2 * b < live ? sums[2 * b] : 0);
            dots_arr[2 * b + 1] = s1 - (2 * b + 1 < live ? sums[2 * b + 1] : 0);
        }
        alignas(32) float fdots[8];
        for (int i = 0; i < 8; ++i) { fdots[i] = float(dots_arr[i]); }
        return _mm256_load_ps(fdots);
    };
    rows_in_lane_order<false>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

template <bool kNl>
void block32_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                  float* y, std::size_t y_stride) {
    constexpr int bytes = kNl ? 18 : 34;
    const auto decode   = [](const std::uint8_t* row, int g, int live, Group& group) {
        for (int i = 0; i < 8; ++i) {
            if (i >= live) {
                group.factor[0][i] = 0.0F;
                group.magnitude[i][0] = group.magnitude[i][1] = _mm_setzero_si128();
                group.sign[i][0] = group.sign[i][1] = _mm_setzero_si128();
                continue;
            }
            const std::uint8_t* block = row + std::size_t(8 * g + i) * bytes;
            __m128i codes_lo, codes_hi;
            if constexpr (kNl) {
                iq4_values(block + 2, codes_lo, codes_hi);
            } else {
                codes_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
                codes_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 18));
            }
            group.magnitude[i][0] = _mm_sign_epi8(codes_lo, codes_lo);
            group.magnitude[i][1] = _mm_sign_epi8(codes_hi, codes_hi);
            group.sign[i][0]      = codes_lo;
            group.sign[i][1]      = codes_hi;
            group.factor[0][i]    = half_at(block);
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m128i u_lo[8], u_hi[8];
        for (int i = 0; i < 8; ++i) {
            const __m128i v_lo = i < live ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 32 * i))
                                          : _mm_setzero_si128();
            const __m128i v_hi = i < live ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 32 * i + 16))
                                          : _mm_setzero_si128();
            u_lo[i] = dot_u8s8_128(group.magnitude[i][0], _mm_sign_epi8(v_lo, group.sign[i][0]));
            u_hi[i] = dot_u8s8_128(group.magnitude[i][1], _mm_sign_epi8(v_hi, group.sign[i][1]));
        }
        __m128i lo0123, lo4567, hi0123, hi4567;
        sum_hadd8(u_lo, lo0123, lo4567);
        sum_hadd8(u_hi, hi0123, hi4567);
        const __m128i tot_lo = _mm_add_epi32(lo0123, hi0123);
        const __m128i tot_hi = _mm_add_epi32(lo4567, hi4567);
        return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(tot_lo)),
                                    _mm_cvtepi32_ps(tot_hi), 1);
    };
    rows_in_lane_order<false>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

} // namespace

void rows_avx(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
              float* y, std::size_t y_stride) {
    if (x.tokens() > kMaxTokens) { return rows_scalar(w, x, row_begin, row_end, y, y_stride); }
    switch (w.format) {
    case QType::GGUF_IQ2_S: return iq_rows<QType::GGUF_IQ2_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_XXS:
        return iq_rows<QType::GGUF_IQ3_XXS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_S: return iq_rows<QType::GGUF_IQ3_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_XS:
        return iq_rows<QType::GGUF_IQ4_XS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q2_0: return q2_rows(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_NL: return block32_rows<true>(w, x, row_begin, row_end, y, y_stride);
    default: return block32_rows<false>(w, x, row_begin, row_end, y, y_stride);
    }
}

// silu.h eight values at a time using AVX 1.0 + SSE4.1 (with emulated FMA): same operations
// in the same order as swiglu_avx2, so the same bits.
void swiglu_avx(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    const __m256 one     = _mm256_set1_ps(1.0F);
    const __m128i lo_126 = _mm_set1_epi32(-126);
    const __m128i hi_127 = _mm_set1_epi32(127);
    std::size_t i        = 0;
    alignas(32) float products[8];
    for (; i + 8 <= count; i += 8) {
        const __m256 x = _mm256_loadu_ps(gate + i);
        const __m256 v = _mm256_xor_ps(x, _mm256_set1_ps(-0.0F)); // e^v, v = -x
        const __m256 n = _mm256_round_ps(_mm256_mul_ps(v, _mm256_set1_ps(1.44269504F)),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = fma8(n, _mm256_set1_ps(-0.693145751953125F), v);
        r        = fma8(n, _mm256_set1_ps(-1.42860677e-6F), r);
        __m256 p = fma8(_mm256_set1_ps(1.9875691500e-4F), r, _mm256_set1_ps(1.3981999507e-3F));
        p        = fma8(p, r, _mm256_set1_ps(8.3334519073e-3F));
        p        = fma8(p, r, _mm256_set1_ps(4.1665795894e-2F));
        p        = fma8(p, r, _mm256_set1_ps(1.6666665459e-1F));
        p        = fma8(p, r, _mm256_set1_ps(5.0000001201e-1F));
        p        = fma8(p, _mm256_mul_ps(r, r), r);
        p        = _mm256_add_ps(p, one);
        // Split int32 exponent scaling into two 128-bit SSE4.1/SSE2 halves (no AVX2).
        const __m256i k     = _mm256_cvttps_epi32(n);
        const __m128i k_lo  = _mm256_castsi256_si128(k);
        const __m128i k_hi  = _mm256_extractf128_si256(k, 1);
        const __m128i c_lo  = _mm_min_epi32(_mm_max_epi32(k_lo, lo_126), hi_127);
        const __m128i c_hi  = _mm_min_epi32(_mm_max_epi32(k_hi, lo_126), hi_127);
        const __m128 p_lo   = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(c_lo, hi_127), 23));
        const __m128 p_hi   = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(c_hi, hi_127), 23));
        const __m256 power  = _mm256_insertf128_ps(_mm256_castps128_ps256(p_lo), p_hi, 1);
        const __m128 b_lo   = _mm_castsi128_ps(_mm_cmpgt_epi32(k_lo, hi_127));
        const __m128 b_hi   = _mm_castsi128_ps(_mm_cmpgt_epi32(k_hi, hi_127));
        const __m256 beyond = _mm256_insertf128_ps(_mm256_castps128_ps256(b_lo), b_hi, 1);
        __m256 e = _mm256_mul_ps(_mm256_mul_ps(p, power),
                                 _mm256_blendv_ps(one, _mm256_set1_ps(2.0F), beyond));
        e = _mm256_blendv_ps(e, _mm256_setzero_ps(),
                             _mm256_cmp_ps(v, _mm256_set1_ps(-87.3365479F), _CMP_LT_OQ));
        e = _mm256_blendv_ps(e, _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000)),
                             _mm256_cmp_ps(v, _mm256_set1_ps(88.7228394F), _CMP_GE_OQ));
        e = _mm256_blendv_ps(e, v, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        const __m256 silu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_store_ps(products, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
        for (int j = 0; j < 8; ++j) { out[i + j] = float_to_bf16(products[j]); }
    }
    for (; i < count; ++i) { out[i] = float_to_bf16(ops::silu(gate[i]) * up[i]); }
}

void fma_avx(const float* a, const float* b, const float* c, float* out, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(out + i, fma8(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i),
                                       _mm256_loadu_ps(c + i)));
    }
    if (i < count) {
        alignas(16) float ta[4]{}, tb[4]{}, tc[4]{}, to[4]{};
        for (; i + 4 <= count; i += 4) {
            _mm_storeu_ps(out + i, fma4(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i),
                                        _mm_loadu_ps(c + i)));
        }
        const std::size_t rem = count - i;
        if (rem > 0) {
            std::memcpy(ta, a + i, rem * sizeof(float));
            std::memcpy(tb, b + i, rem * sizeof(float));
            std::memcpy(tc, c + i, rem * sizeof(float));
            _mm_store_ps(to, fma4(_mm_load_ps(ta), _mm_load_ps(tb), _mm_load_ps(tc)));
            std::memcpy(out + i, to, rem * sizeof(float));
        }
    }
}

} // namespace ninfer::ops::gguf_cpu_detail
