// The fast ggml-cpu arithmetic of the GGUF CPU experts (gguf_expert_cpu_ggml): q8_K / q8_0
// activations through ggml's AVX2 vecdot kernels, against (1) an FP64 oracle from independently
// decoded stored blocks and (2) the exact q8_1 path, at the Flash-Next expert shapes, for every
// format the fast path supports. The q8 activation bounds the error: relative L2 under 3% and
// the worst output within 10% of the oracle's peak, like the exact path's int8 activation.
#include "ops/moe_experts/gguf_expert_cpu.h"
#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"
#include "quants.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

std::uint16_t half_bits(float value) {
    const auto word = std::bit_cast<std::uint32_t>(value);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    const std::uint32_t mantissa = ((word & 0x7fffff) + 0x1000) >> 13;
    return static_cast<std::uint16_t>((exponent << 10) + mantissa);
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

struct Matrix {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // decoded independently from the stored bytes
};

// One stored block of `format` to its FP64 values. The i-quant decodes follow ggml-quants.c's
// dequantize_row_* (llama.cpp a25c9865, MIT); the exact path's oracles use the same formulas.
void reference_block(QType format, const std::uint8_t* bytes, double* y) {
    if (format == QType::GGUF_Q8_0) {
        const auto* x     = reinterpret_cast<const block_q8_0*>(bytes);
        const double d    = exact_half(x->d);
        for (int j = 0; j < QK8_0; ++j) { y[j] = d * static_cast<std::int8_t>(x->qs[j]); }
        return;
    }
    if (format == QType::GGUF_IQ4_NL) {
        const auto* x  = reinterpret_cast<const block_iq4_nl*>(bytes);
        const double d = exact_half(x->d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j]              = d * kvalues_iq4nl[x->qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * kvalues_iq4nl[x->qs[j] >> 4];
        }
        return;
    }
    if (format == QType::GGUF_IQ4_XS) {
        const auto* x      = reinterpret_cast<const block_iq4_xs*>(bytes);
        const std::uint8_t* qs = x->qs;
        const double d     = exact_half(x->d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0]  = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
        return;
    }
    if (format == QType::GGUF_IQ2_XS) {
        const auto* x      = reinterpret_cast<const block_iq2_xs*>(bytes);
        const double d     = exact_half(x->d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const double db[2] = {d * (0.5 + (x->scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (x->scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const auto* grid = reinterpret_cast<const std::uint8_t*>(
                    iq2xs_grid + (x->qs[4 * ib32 + l] & 511));
                const std::uint8_t signs = ksigns_iq2xs[x->qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
        }
        return;
    }
    if (format == QType::GGUF_IQ2_XXS) {
        const auto* x = reinterpret_cast<const block_iq2_xxs*>(bytes);
        const double d = exact_half(x->d);
        std::uint32_t aux32[2];
        const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(aux32, x->qs + 4 * ib32, 2 * sizeof(std::uint32_t));
            const double db = d * (0.5 + (aux32[1] >> 28)) * 0.25;
            for (int l = 0; l < 4; ++l) {
                const auto* grid = reinterpret_cast<const std::uint8_t*>(iq2xxs_grid + aux8[l]);
                const std::uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
        }
        return;
    }
    if (format == QType::GGUF_IQ2_S) {
        const auto* x             = reinterpret_cast<const block_iq2_s*>(bytes);
        const double d            = exact_half(x->d);
        const std::uint8_t* qs    = x->qs;
        const std::uint8_t* qh    = x->qh;
        const std::uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const double db[2] = {d * (0.5 + (x->scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (x->scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const double dl = db[l / 2];
                const auto* grid = reinterpret_cast<const std::uint8_t*>(
                    iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
        return;
    }
    if (format == QType::GGUF_IQ3_XXS) {
        const auto* x       = reinterpret_cast<const block_iq3_xxs*>(bytes);
        const double d      = exact_half(x->d);
        const std::uint8_t* qs = x->qs;
        const std::uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(aux32));
            const double db = d * (0.5 + (aux32 >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l + 0]);
                const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                    iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 8;
        }
        return;
    }
    if (format == QType::GGUF_IQ3_S) {
        const auto* x             = reinterpret_cast<const block_iq3_s*>(bytes);
        const double d            = exact_half(x->d);
        const std::uint8_t* qs    = x->qs;
        const std::uint8_t* qh    = x->qh;
        const std::uint8_t* signs = x->signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const double db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
            const double db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
            for (int half = 0; half < 2; ++half) {
                const double db = half == 0 ? db1 : db2;
                for (int l = 0; l < 4; ++l) {
                    const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                        iq3s_grid + (qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)));
                    const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                        iq3s_grid + (qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                        y[j + 4] = db * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
        return;
    }
    throw std::runtime_error("reference_block: unhandled format");
}

Matrix make(QType format, int rows, int k, std::mt19937& rng) {
    const auto [group, block] = gguf_block_shape(format);
    Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
    std::uniform_real_distribution<float> scale(
        format == QType::GGUF_Q8_0 ? 0.002F : 0.0005F,
        format == QType::GGUF_Q8_0 ? 0.05F : 0.004F);
    m.values.resize(std::size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d    = half_bits(scale(rng));
            const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
            std::memcpy(p, &signed_d, 2);
            reference_block(format, p, m.values.data() + std::size_t(r) * k + std::size_t(b) * group);
        }
    }
    return m;
}

GgmlActLayout layout_for(QType format) {
    return (format == QType::GGUF_IQ4_NL || format == QType::GGUF_Q8_0) ? GgmlActLayout::Q8_0
                                                                        : GgmlActLayout::Q8K;
}

void check(const Matrix& m, int tokens, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(std::size_t(tokens) * m.k);
    for (auto& v : x) { v = bf16(normal(rng)); }
    std::vector<double> oracle(std::size_t(tokens) * m.rows);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0;
            for (int i = 0; i < m.k; ++i) {
                sum += m.values[std::size_t(r) * m.k + i] *
                       double(std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + i]) << 16));
            }
            oracle[std::size_t(t) * m.rows + r] = sum;
        }
    }
    const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};
    // The exact path where it covers the format (IQ2_XXS and IQ2_XS are fast-path only):
    // the reference product, and the yardstick for the fast path's drift.
    std::vector<float> exact(oracle.size());
    if (gguf_cpu_supported(m.format)) {
        GgufCpuActivation a(16, m.k);
        a.prepare_bf16(m.format, x.data(), m.k, tokens);
        gguf_cpu_rows(w, a, 0, m.rows, exact.data(), m.rows, GgufCpuBackend::Scalar);
    }
    // The fast path.
    GgmlCpuActivation fa(16, m.k);
    fa.prepare_bf16(layout_for(m.format), x.data(), m.k, tokens);
    std::vector<float> y(oracle.size()), again(oracle.size());
    // Two row ranges, as the workers split them, then the whole matrix again.
    gguf_cpu_rows_ggml(w, fa, 0, m.rows / 3, y.data(), m.rows);
    gguf_cpu_rows_ggml(w, fa, m.rows / 3, m.rows, y.data(), m.rows);
    gguf_cpu_rows_ggml(w, fa, 0, m.rows, again.data(), m.rows);
    double error = 0, norm = 0, peak = 0, worst = 0, diff = 0, diff_norm = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        require(std::isfinite(y[i]), "nonfinite output");
        require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(again[i]),
                "a row split changed the result or a repeat differed");
        error += (double(y[i]) - oracle[i]) * (double(y[i]) - oracle[i]);
        norm  += oracle[i] * oracle[i];
        peak   = std::max(peak, std::abs(oracle[i]));
        worst  = std::max(worst, std::abs(double(y[i]) - oracle[i]));
        if (gguf_cpu_supported(m.format)) {
            diff      += (double(y[i]) - exact[i]) * (double(y[i]) - exact[i]);
            diff_norm += exact[i] * exact[i];
        }
    }
    const double relative = std::sqrt(error / norm);
    const double vs_exact = gguf_cpu_supported(m.format) ? std::sqrt(diff / diff_norm) : 0;
    std::cout << "ggml " << int(m.format) << " " << m.rows << "x" << m.k << " T=" << tokens << ": "
              << "relative L2 " << relative << ", max error/peak " << worst / peak << ", vs exact "
              << vs_exact << '\n';
    require(relative < 0.03 && worst / peak < 0.1, "error exceeds the q8-activation bound");
    require(vs_exact < 0.05, "the fast path strays from the exact path");
}

void check_bf16_quantizers_bit_exact_vs_ref(std::mt19937& rng) {
    auto verify_q8_0 = [](const std::vector<std::uint16_t>& bf, const char* label) {
        const int k = static_cast<int>(bf.size());
        const int nb = k / QK8_0;
        std::vector<float> fp32(k);
        for (int i = 0; i < k; ++i) {
            fp32[i] = std::bit_cast<float>(std::uint32_t(bf[i]) << 16);
        }
        std::vector<block_q8_0> ref(nb), got(nb);
        std::memset(ref.data(), 0xA5, ref.size() * sizeof(block_q8_0));
        std::memset(got.data(), 0x5A, got.size() * sizeof(block_q8_0));
        quantize_row_q8_0_ref(fp32.data(), ref.data(), k);
        ggml_quantize_bf16_q8_0(bf.data(), got.data(), k);
        for (int b = 0; b < nb; ++b) {
            require(std::memcmp(&ref[b], &got[b], sizeof(block_q8_0)) == 0,
                    std::string("q8_0 byte mismatch vs ref: ") + label);
        }
    };
    auto verify_q8_K = [](const std::vector<std::uint16_t>& bf, const char* label,
                          bool expect_zero_bsums) {
        const int k = static_cast<int>(bf.size());
        const int nb = k / QK_K;
        std::vector<float> fp32(k);
        for (int i = 0; i < k; ++i) {
            fp32[i] = std::bit_cast<float>(std::uint32_t(bf[i]) << 16);
        }
        std::vector<block_q8_K> ref(nb), got(nb);
        // Pre-fill with 0xA5 so we observe that all-zero blocks zero bsums in AVX while _ref
        // intentionally leaves bsums untouched.
        std::memset(ref.data(), expect_zero_bsums ? 0xA5 : 0x00, ref.size() * sizeof(block_q8_K));
        std::memset(got.data(), 0x5A, got.size() * sizeof(block_q8_K));
        quantize_row_q8_K_ref(fp32.data(), ref.data(), k);
        ggml_quantize_bf16_q8_K(bf.data(), got.data(), k);
        for (int b = 0; b < nb; ++b) {
            if (expect_zero_bsums) {
                require(got[b].d == 0.0F, "all-zero q8_K d not zero");
                for (int j = 0; j < QK_K; ++j) { require(got[b].qs[j] == 0, "all-zero q8_K qs"); }
                for (int j = 0; j < QK_K / 16; ++j) {
                    require(got[b].bsums[j] == 0, "all-zero q8_K bsums not zeroed by AVX");
                }
            } else {
                require(std::memcmp(&ref[b], &got[b], sizeof(block_q8_K)) == 0,
                        std::string("q8_K byte mismatch vs ref: ") + label);
            }
        }
    };

    // 1. All-zero blocks.
    verify_q8_0(std::vector<std::uint16_t>(32, 0), "all-zero");
    verify_q8_K(std::vector<std::uint16_t>(256, 0), "all-zero", /*expect_zero_bsums=*/true);

    // 2. Two equal magnitudes with opposite signs (rule of the first maximum).
    {
        std::vector<std::uint16_t> ties(256, bf16(1.0F));
        ties[0] = bf16(5.0F);
        ties[5] = bf16(-5.0F);
        verify_q8_0(ties, "equal-abs-opp-sign");
        verify_q8_K(ties, "equal-abs-opp-sign", false);
        ties[0] = bf16(-5.0F);
        ties[5] = bf16(5.0F);
        verify_q8_K(ties, "equal-abs-opp-sign-neg-first", false);
    }

    // 3. Exact .5 ties in v/d (q8_0: roundf ties-away) and iscale*x (q8_K: nearest_int ties-even).
    {
        std::vector<std::uint16_t> ties(256, 0);
        ties[0]  = bf16(127.0F);
        ties[1]  = bf16(0.5F);
        ties[2]  = bf16(-0.5F);
        ties[3]  = bf16(1.5F);
        ties[4]  = bf16(-1.5F);
        ties[5]  = bf16(2.5F);
        ties[6]  = bf16(-2.5F);
        ties[7]  = bf16(63.5F);
        ties[8]  = bf16(-63.5F);
        ties[9]  = 0x8000U; // -0.0
        verify_q8_0(ties, "exact-halves");
        verify_q8_K(ties, "exact-halves", false);
    }

    // 4. BF16 subnormal (1e-40 -> 0x0001) alongside 1.0, and 1e30.
    {
        std::vector<std::uint16_t> sub(256, bf16(1.0F));
        sub[3] = 0x0001U;
        sub[4] = 0x8001U;
        verify_q8_0(sub, "subnormal-mixed");
        verify_q8_K(sub, "subnormal-mixed", false);
        std::vector<std::uint16_t> big(256, bf16(1e30F));
        big[1] = bf16(-1e30F);
        verify_q8_0(big, "1e30");
        verify_q8_K(big, "1e30", false);
    }

    // 5. Random sweep across k=32/256/640/2560.
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (const int k : {32, 256, 640, 2560}) {
        std::vector<std::uint16_t> x(k);
        for (auto& v : x) { v = bf16(normal(rng)); }
        verify_q8_0(x, "random-q8_0");
        if (k % 256 == 0) { verify_q8_K(x, "random-q8_K", false); }
    }
}

} // namespace

int main() {
    // T-03: on x86_64 builds with matching ISA (AVX+F16C when AVX2=OFF, AVX2+FMA when AVX2=ON),
    // gguf_ggml_cpu_available() must report true so the test never silently skips.
#if defined(__x86_64__) || defined(_M_X64)
    const bool isa_ok = NINFER_GGML_CPU_REQUIRES_AVX2
                            ? gguf_cpu_backend_available(GgufCpuBackend::Avx2)
                            : gguf_cpu_backend_available(GgufCpuBackend::Avx);
    require(gguf_ggml_cpu_available() == isa_ok,
            "gguf_ggml_cpu_available() disagreed with host ISA and NINFER_GGML_CPU_AVX2");
#endif
    if (!gguf_ggml_cpu_available()) {
        std::cout << "gguf_expert_cpu_ggml: skipped (CPU lacks required instruction set)\n";
        return 77;
    }
    try {
        std::mt19937 rng(20261030);
        check_bf16_quantizers_bit_exact_vs_ref(rng);
        // The gate and up projections: k = 2560 (ten 256-value blocks).
        for (const QType format : {QType::GGUF_IQ2_XXS, QType::GGUF_IQ2_XS, QType::GGUF_IQ2_S,
                                   QType::GGUF_IQ3_XXS, QType::GGUF_IQ3_S, QType::GGUF_IQ4_XS,
                                   QType::GGUF_IQ4_NL, QType::GGUF_Q8_0}) {
            require(gguf_ggml_supported(format), "a format is not marked supported");
            const Matrix gate = make(format, 640, 2560, rng);
            for (const int tokens : {1, 2, 4, 8, 9}) { check(gate, tokens, rng); }
            // The down projection: k = 640, a multiple of 32 but not 256 (Q8_0 activations only).
            if (layout_for(format) == GgmlActLayout::Q8_0) {
                const Matrix down = make(format, 2560, 640, rng);
                for (const int tokens : {1, 4, 8}) { check(down, tokens, rng); }
            }
        }
        // Uncovered formats refuse instead of silently falling back.
        for (const QType format : {QType::GGUF_Q2_0, QType::GGUF_Q4_K, QType::GGUF_Q6_K}) {
            require(!gguf_ggml_supported(format), "an uncovered format is marked supported");
        }
        bool refused = false;
        try {
            GgmlCpuActivation a(1, 2560);
            std::vector<std::uint16_t> x(2560 * 2);
            a.prepare_bf16(GgmlActLayout::Q8K, x.data(), 2560, 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "an activation above its capacity was accepted");
        refused = false;
        try { GgmlCpuActivation a(1, 100); } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "a k that is not a multiple of 32 was accepted");
        refused = false;
        try {
            GgmlCpuActivation a(1, 640);
            std::vector<std::uint16_t> x(640 * 2);
            a.prepare_bf16(GgmlActLayout::Q8K, x.data(), 640, 1);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "a Q8_K activation with k not a multiple of 256 was accepted");
        // A non-finite activation must be rejected before the fast ggml quantizer, which would
        // otherwise derive a NaN block scale and emit garbage codes (the exact path throws for the
        // same input, so the two backends must agree).
        auto expect_nonfinite = [&](const std::string& kind, auto&& call) {
            bool threw = false;
            std::string message;
            try { call(); } catch (const std::runtime_error& error) {
                threw = true;
                message = error.what();
            }
            require(threw && message.find("nonfinite") != std::string::npos,
                    kind + " activation was not rejected as nonfinite");
        };
        auto finite_row = [&] { return std::vector<std::uint16_t>(2560, 0x3F00); };  // all 1.0
        GgmlCpuActivation single(1, 2560);
        single.prepare_bf16_start(GgmlActLayout::Q8K, 1);
        {
            auto row = finite_row();
            row[1234] = 0x7F81;  // NaN
            expect_nonfinite("a NaN", [&] { single.prepare_bf16_token(0, row.data()); });
        }
        {
            auto row = finite_row();
            row[2559] = 0x7F81;  // NaN at k-1
            expect_nonfinite("a tail NaN", [&] { single.prepare_bf16_token(0, row.data()); });
        }
        {
            auto row = finite_row();
            row[2552] = 0xFF80;  // -Inf in the last 8-element vector
            expect_nonfinite("a last-vec8 Inf", [&] { single.prepare_bf16_token(0, row.data()); });
        }
        {
            auto row = finite_row();
            row[0] = 0x7F80;  // +Inf
            expect_nonfinite("an Inf", [&] { single.prepare_bf16_token(0, row.data()); });
        }
        {
            // Finite extremes: 1e-40 makes 127/amax overflow to Inf in both Q8_K and Q8_0;
            // +/-1e30 overflows Q8_0's FP16 block scale to Inf.
            std::vector<std::uint16_t> tiny_row(2560, 0);
            tiny_row[2559] = 0x0001; // ~9.18e-41 (< 1e-40)
            expect_nonfinite("a tiny Q8_K block", [&] { single.prepare_bf16_token(0, tiny_row.data()); });

            GgmlCpuActivation single_q80(1, 640);
            single_q80.prepare_bf16_start(GgmlActLayout::Q8_0, 1);
            std::vector<std::uint16_t> tiny_q80(640, 0);
            tiny_q80[639] = 0x0001;
            expect_nonfinite("a tiny Q8_0 block", [&] { single_q80.prepare_bf16_token(0, tiny_q80.data()); });

            std::vector<std::uint16_t> huge_q80(640, bf16(1.0F));
            huge_q80[639] = bf16(1e30F);
            expect_nonfinite("a +1e30 Q8_0 block", [&] { single_q80.prepare_bf16_token(0, huge_q80.data()); });
            huge_q80[639] = bf16(-1e30F);
            expect_nonfinite("a -1e30 Q8_0 block", [&] { single_q80.prepare_bf16_token(0, huge_q80.data()); });
        }
        {
            std::vector<std::uint16_t> both(2 * 2560, 0x3F00);
            both[2560 + 7] = 0xFFC0;  // -Inf in the second token
            GgmlCpuActivation batch(2, 2560);
            expect_nonfinite("a batch Inf", [&] {
                batch.prepare_bf16(GgmlActLayout::Q8K, both.data(), 2560, 2);
            });
        }
        std::cout << "gguf_expert_cpu_ggml: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_expert_cpu_ggml: " << error.what() << '\n';
        return 1;
    }
}
