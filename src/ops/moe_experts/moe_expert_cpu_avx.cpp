#include "ops/moe_experts/cpu_projection.h"
#include <immintrin.h>
#include <cstring>

namespace ninfer::ops::cpu_expert_detail {
namespace {
struct AvxDot {
    static int integer(const std::int8_t* weights, const std::int8_t* input) {
        // 32 values across two 128-bit SSSE3/SSE2 pairs.
        const auto w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights));
        const auto w1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + 16));
        const auto x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
        const auto x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 16));

        const auto pairs0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
        const auto pairs1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));

        const auto ones16 = _mm_set1_epi16(1);
        const auto sums0 = _mm_madd_epi16(pairs0, ones16);
        const auto sums1 = _mm_madd_epi16(pairs1, ones16);

        auto sum = _mm_add_epi32(sums0, sums1);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
    }

    static float floating(const std::int8_t* weights, const float* input) {
        auto sum = _mm256_setzero_ps();
        for (int start = 0; start < 32; start += 8) {
            std::int32_t c_lo, c_hi;
            std::memcpy(&c_lo, weights + start, 4);
            std::memcpy(&c_hi, weights + start + 4, 4);
            const auto i_lo = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(c_lo));
            const auto i_hi = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(c_hi));
            const auto f_lo = _mm_cvtepi32_ps(i_lo);
            const auto f_hi = _mm_cvtepi32_ps(i_hi);
            const auto w = _mm256_insertf128_ps(_mm256_castps128_ps256(f_lo), f_hi, 1);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(w, _mm256_loadu_ps(input + start)));
        }
        auto halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        halves = _mm_hadd_ps(halves, halves);
        return _mm_cvtss_f32(_mm_hadd_ps(halves, halves));
    }
};
} // namespace

void project_avx(const Weight& w, const Activations& x, bool a8, float* y) {
    project<AvxDot>(w, x, a8, y);
}
} // namespace ninfer::ops::cpu_expert_detail
