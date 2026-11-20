// Direct FP64-oracle, non-finite preservation, degenerate-size and AVX1 equivalence checks
// for Qwen3.5 CPU vision kernels (portable, AVX1 and AVX2).
#include "models/qwen3_5/execution/vision_cpu.h"
#include "models/qwen3_5/execution/vision_cpu_kernels.h"
#include "models/qwen3_5/load/vision_cpu.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace cv = ninfer::models::qwen3_5::execution::cpu_vision;
using ninfer::models::qwen3_5::CpuVisionWeights;
using ninfer::models::qwen3_5::VisionItemControl;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

void test_gemm_packed_bias(const cv::Kernels& kernels, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    constexpr float kPoison = -777.25F;
    for (const int n : {1, 2, 5, 6, 7, 12, 13, 35}) {
        for (const int tokens : {1, 15, 16, 17, 31, 257}) {
            const int k                  = 19;
            const std::size_t w_stride   = static_cast<std::size_t>(k) + 5;
            const std::size_t x_stride   = static_cast<std::size_t>(tokens) + 3;
            const std::size_t y_stride   = static_cast<std::size_t>(tokens) + 7;
            std::vector<float> w(static_cast<std::size_t>(n) * w_stride, kPoison);
            std::vector<float> x(static_cast<std::size_t>(k) * x_stride, kPoison);
            std::vector<float> bias(static_cast<std::size_t>(n));
            for (int r = 0; r < n; ++r) {
                bias[r] = normal(rng);
                for (int i = 0; i < k; ++i) { w[static_cast<std::size_t>(r) * w_stride + i] = normal(rng); }
            }
            for (int i = 0; i < k; ++i) {
                for (int t = 0; t < tokens; ++t) { x[static_cast<std::size_t>(i) * x_stride + t] = normal(rng); }
            }
            const int panels = (tokens + 15) / 16;
            std::vector<float> packed(static_cast<std::size_t>(panels) * k * 16, kPoison);
            for (int p = 0; p < panels; ++p) {
                const int tok   = p * 16;
                const int width = std::min(16, tokens - tok);
                kernels.pack_panel(x.data(), x_stride, k, tok, width,
                                   packed.data() + static_cast<std::size_t>(p) * k * 16);
            }
            std::vector<float> y(static_cast<std::size_t>(n + 2) * y_stride + 9, kPoison);
            kernels.gemm_packed_bias(w.data(), w_stride, bias.data(), n, k, packed.data(), tokens,
                                     y.data(), y_stride);
            double worst = 0.0;
            bool kept    = true;
            for (int r = 0; r < n; ++r) {
                for (std::size_t t = 0; t < y_stride; ++t) {
                    const float got = y[static_cast<std::size_t>(r) * y_stride + t];
                    if (t >= static_cast<std::size_t>(tokens)) {
                        kept &= got == kPoison;
                        continue;
                    }
                    double want = bias[r], scale = std::abs(double(bias[r]));
                    for (int i = 0; i < k; ++i) {
                        const double prod =
                            double(w[static_cast<std::size_t>(r) * w_stride + i]) *
                            double(x[static_cast<std::size_t>(i) * x_stride + t]);
                        want += prod;
                        scale += std::abs(prod);
                    }
                    worst = std::max(worst, std::abs(double(got) - want) / std::max(scale, 1e-12));
                }
            }
            for (std::size_t i = static_cast<std::size_t>(n) * y_stride; i < y.size(); ++i) {
                kept &= y[i] == kPoison;
            }
            const std::string label = std::string(kernels.name) + " gemm_packed_bias " +
                                      std::to_string(n) + "x" + std::to_string(k) + "x" +
                                      std::to_string(tokens);
            expect(worst <= 1e-5, label + ": error " + std::to_string(worst));
            expect(kept, label + ": overwrote sentinel past tokens/rows");
        }
    }
}

void test_layer_norm_cols(const cv::Kernels& kernels, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 2.0F);
    constexpr std::int32_t kFeatures = 72;
    constexpr float kEpsilon         = 1e-6F;
    constexpr float kPoison          = -555.5F;
    for (const std::int32_t count : {1, 7, 8, 9, 71, 72, 144}) {
        const std::int32_t first  = 3;
        const std::size_t stride  = static_cast<std::size_t>(first + count + 5);
        std::vector<float> x(static_cast<std::size_t>(kFeatures) * stride, kPoison);
        std::vector<float> y(static_cast<std::size_t>(kFeatures) * stride + 7, kPoison);
        std::vector<float> weight(kFeatures), bias(kFeatures);
        for (std::int32_t f = 0; f < kFeatures; ++f) {
            weight[f] = normal(rng);
            bias[f]   = normal(rng);
            for (std::int32_t c = 0; c < count; ++c) {
                x[static_cast<std::size_t>(f) * stride + first + c] = normal(rng);
            }
        }
        kernels.layer_norm_cols(x.data(), stride, kFeatures, weight.data(), bias.data(), first,
                                count, kEpsilon, y.data());
        double worst = 0.0;
        bool kept    = true;
        for (std::int32_t c = 0; c < count; ++c) {
            double mean = 0.0;
            for (std::int32_t f = 0; f < kFeatures; ++f) {
                mean += double(x[static_cast<std::size_t>(f) * stride + first + c]);
            }
            mean /= double(kFeatures);
            double var = 0.0;
            for (std::int32_t f = 0; f < kFeatures; ++f) {
                const double d = double(x[static_cast<std::size_t>(f) * stride + first + c]) - mean;
                var += d * d;
            }
            const double inv = 1.0 / std::sqrt(var / double(kFeatures) + double(kEpsilon));
            for (std::int32_t f = 0; f < kFeatures; ++f) {
                const double want =
                    (double(x[static_cast<std::size_t>(f) * stride + first + c]) - mean) * inv *
                        double(weight[f]) +
                    double(bias[f]);
                const float got = y[static_cast<std::size_t>(f) * stride + first + c];
                worst           = std::max(worst, std::abs(double(got) - want) /
                                                      std::max(1.0, std::abs(want)));
            }
        }
        for (std::int32_t f = 0; f < kFeatures; ++f) {
            for (std::size_t c = 0; c < stride; ++c) {
                if (c < static_cast<std::size_t>(first) ||
                    c >= static_cast<std::size_t>(first + count)) {
                    kept &= y[static_cast<std::size_t>(f) * stride + c] == kPoison;
                }
            }
        }
        for (std::size_t i = static_cast<std::size_t>(kFeatures) * stride; i < y.size(); ++i) {
            kept &= y[i] == kPoison;
        }
        expect(worst <= 1e-4, std::string(kernels.name) + " layer_norm_cols count=" +
                                  std::to_string(count) + ": error " + std::to_string(worst));
        expect(kept, std::string(kernels.name) + " layer_norm_cols overwrote sentinel");
    }
}

void test_rope_and_gelu(const cv::Kernels& kernels, std::mt19937& rng) {
    std::uniform_real_distribution<float> angle_dist(-3.14F, 3.14F);
    std::normal_distribution<float> normal(0.0F, 1.5F);
    for (const std::int32_t tokens : {1, 7, 8, 9, 31}) {
        std::vector<float> first(tokens), second(tokens), cosines(tokens), sines(tokens);
        for (std::int32_t t = 0; t < tokens; ++t) {
            first[t]       = normal(rng);
            second[t]      = normal(rng);
            const float a  = angle_dist(rng);
            cosines[t]     = std::cos(a);
            sines[t]       = std::sin(a);
        }
        const auto f0 = first, s0 = second;
        kernels.rope_pair(first.data(), second.data(), cosines.data(), sines.data(), tokens);
        double worst = 0.0;
        for (std::int32_t t = 0; t < tokens; ++t) {
            const double wf = double(f0[t]) * double(cosines[t]) - double(s0[t]) * double(sines[t]);
            const double ws = double(s0[t]) * double(cosines[t]) + double(f0[t]) * double(sines[t]);
            worst = std::max(worst, std::abs(double(first[t]) - wf));
            worst = std::max(worst, std::abs(double(second[t]) - ws));
        }
        expect(worst <= 1e-6, std::string(kernels.name) + " rope_pair error " + std::to_string(worst));
    }

    std::uniform_real_distribution<float> gate_dist(-12.0F, 12.0F);
    const std::size_t count = 1003;
    std::vector<float> g(count);
    for (std::size_t i = 0; i < count; ++i) {
        g[i] = i < 40 ? static_cast<float>(i) * 1e-3F - 0.02F : gate_dist(rng);
    }
    const auto g_in = g;
    kernels.gelu_tanh(g.data(), count);
    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double x = g_in[i];
        const double want =
            0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        worst = std::max(worst, std::abs(double(g[i]) - want) / (0.5 * std::abs(x) + 1e-6));
    }
    expect(worst <= 1e-5, std::string(kernels.name) + " gelu_tanh error " + std::to_string(worst));
}

void test_segment_attention(const cv::Kernels& kernels, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 0.5F);
    constexpr std::int32_t kHeadDim = 72;
    constexpr std::int32_t kHeads   = 2;
    constexpr std::int32_t kHidden  = kHeads * kHeadDim;
    constexpr float kPoison         = -999.5F;
    const float scale               = static_cast<float>(1.0 / std::sqrt(double(kHeadDim)));

    for (const std::int32_t segment : {1, 8, 9, 16, 17, 256, 257, 273}) {
        for (const std::int32_t qn : {1, 4, std::min(segment, 33)}) {
            const std::int32_t s0       = 4;
            const std::int32_t q0       = s0;
            const std::int32_t row_base = kHeadDim; // head 1
            const std::size_t qkv_stride = static_cast<std::size_t>(s0 + segment + 5);
            const std::size_t out_stride = static_cast<std::size_t>(s0 + segment + 7);
            std::vector<float> qkv(static_cast<std::size_t>(3 * kHidden) * qkv_stride, kPoison);
            for (std::int32_t r = 0; r < 3 * kHidden; ++r) {
                for (std::int32_t t = 0; t < segment; ++t) {
                    qkv[static_cast<std::size_t>(r) * qkv_stride + s0 + t] = normal(rng);
                }
            }
            std::vector<float> scores(static_cast<std::size_t>(qn) * segment, kPoison);
            std::vector<float> out(static_cast<std::size_t>(kHidden) * out_stride + 11, kPoison);
            kernels.segment_attention_block(qkv.data(), qkv_stride, kHidden, row_base, s0, q0, qn,
                                            segment, scale, scores.data(), out.data(), out_stride);

            double worst = 0.0;
            bool kept    = true;
            for (std::int32_t q = 0; q < qn; ++q) {
                std::vector<double> s_ref(static_cast<std::size_t>(segment));
                double max_s = -std::numeric_limits<double>::infinity();
                for (std::int32_t k = 0; k < segment; ++k) {
                    double dot = 0.0;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        const double qv =
                            qkv[static_cast<std::size_t>(row_base + d) * qkv_stride + q0 + q];
                        const double kv =
                            qkv[static_cast<std::size_t>(kHidden + row_base + d) * qkv_stride + s0 +
                                k];
                        dot += (qv * double(scale)) * kv;
                    }
                    s_ref[k] = dot;
                    max_s    = std::max(max_s, dot);
                }
                double sum = 0.0;
                for (double& v : s_ref) {
                    v = std::exp(v - max_s);
                    sum += v;
                }
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    double want = 0.0;
                    for (std::int32_t k = 0; k < segment; ++k) {
                        const double vv =
                            qkv[static_cast<std::size_t>(2 * kHidden + row_base + d) * qkv_stride +
                                s0 + k];
                        want += (s_ref[k] / sum) * vv;
                    }
                    const float got =
                        out[static_cast<std::size_t>(row_base + d) * out_stride + q0 + q];
                    worst = std::max(worst, std::abs(double(got) - want) /
                                                std::max(1.0, std::abs(want)));
                }
            }
            for (std::int32_t r = 0; r < kHidden; ++r) {
                for (std::size_t c = 0; c < out_stride; ++c) {
                    const bool in_target = r >= row_base && r < row_base + kHeadDim &&
                                           c >= static_cast<std::size_t>(q0) &&
                                           c < static_cast<std::size_t>(q0 + qn);
                    if (!in_target) { kept &= out[static_cast<std::size_t>(r) * out_stride + c] == kPoison; }
                }
            }
            for (std::size_t i = static_cast<std::size_t>(kHidden) * out_stride; i < out.size(); ++i) {
                kept &= out[i] == kPoison;
            }
            expect(worst <= 1e-4, std::string(kernels.name) + " segment_attention seg=" +
                                      std::to_string(segment) + " qn=" + std::to_string(qn) +
                                      ": error " + std::to_string(worst));
            expect(kept, std::string(kernels.name) + " segment_attention overwrote sentinel");
        }
    }
}

void test_nonfinite_preservation(const cv::Kernels& kernels) {
    // 1. gelu_tanh must preserve NaN.
    {
        std::vector<float> v(17, 1.0F);
        v[0]  = std::numeric_limits<float>::quiet_NaN();
        v[9]  = std::numeric_limits<float>::quiet_NaN();
        v[16] = std::numeric_limits<float>::quiet_NaN();
        kernels.gelu_tanh(v.data(), v.size());
        expect(std::isnan(v[0]) && std::isnan(v[9]) && std::isnan(v[16]),
               std::string(kernels.name) + " gelu_tanh dropped NaN");
    }

    // 2. segment_attention_block must preserve NaN and +Inf-Inf=NaN in scores rather than
    //    clamping them to exp(-87.3) or dropping NaN inside _mm256_max_ps(vmax, ...).
    constexpr std::int32_t kHeadDim = 72;
    constexpr std::int32_t kHidden  = kHeadDim;
    const float nan                 = std::numeric_limits<float>::quiet_NaN();
    const float inf                 = std::numeric_limits<float>::infinity();
    for (const std::int32_t segment : {9, 273}) {
        for (const std::int32_t bad_key : {0, 5, segment - 1}) {
            for (const float bad_val : {nan, inf}) {
                std::vector<float> qkv(static_cast<std::size_t>(3 * kHidden) * segment, 0.0F);
                // Set Q[d=0, q=0] = 1.0, K[d=0, k=bad_key] = bad_val, V[d=0, *] = 1.0.
                // Score[bad_key] is NaN (or +Inf, giving +Inf - +Inf = NaN in exp(score - max)).
                qkv[0]                                                  = 1.0F;
                qkv[static_cast<std::size_t>(kHidden) * segment + bad_key] = bad_val;
                for (std::int32_t k = 0; k < segment; ++k) {
                    qkv[static_cast<std::size_t>(kHidden) * segment + k] =
                        (k == bad_key) ? bad_val : 1.0F;
                    qkv[static_cast<std::size_t>(2 * kHidden) * segment + k] = 1.0F;
                }
                std::vector<float> scores(static_cast<std::size_t>(segment), 0.0F);
                std::vector<float> out(static_cast<std::size_t>(kHidden) * segment, 0.0F);
                kernels.segment_attention_block(qkv.data(), static_cast<std::size_t>(segment),
                                                kHidden, 0, 0, 0, 1, segment, 1.0F, scores.data(),
                                                out.data(), static_cast<std::size_t>(segment));
                expect(std::isnan(out[0]),
                       std::string(kernels.name) + " segment_attention swallowed nonfinite score" +
                           " (seg=" + std::to_string(segment) + " key=" + std::to_string(bad_key) +
                           ")");
            }
        }
    }
}

void test_encode_isfinite_gate() {
    CpuVisionWeights w;
    w.hidden               = 72;
    w.heads                = 1;
    w.intermediate         = 16;
    w.patch_width          = 8;
    w.merge_unit           = 1;
    w.output_hidden        = 16;
    w.position_rows        = 1;
    w.patch_embedding.assign(static_cast<std::size_t>(w.hidden) * w.patch_width, 0.01F);
    w.patch_embedding_bias.assign(w.hidden, 0.0F);
    w.position_embedding.assign(w.hidden, 0.0F);
    CpuVisionWeights::Layer layer;
    layer.norm1_weight.assign(w.hidden, 1.0F);
    layer.norm1_bias.assign(w.hidden, 0.0F);
    layer.qkv.assign(static_cast<std::size_t>(3 * w.hidden) * w.hidden, 0.01F);
    layer.qkv_bias.assign(3 * w.hidden, 0.0F);
    layer.output.assign(static_cast<std::size_t>(w.hidden) * w.hidden, 0.01F);
    layer.output_bias.assign(w.hidden, 0.0F);
    layer.norm2_weight.assign(w.hidden, 1.0F);
    layer.norm2_bias.assign(w.hidden, 0.0F);
    layer.fc1.assign(static_cast<std::size_t>(w.intermediate) * w.hidden, 0.01F);
    layer.fc1_bias.assign(w.intermediate, 0.0F);
    layer.fc2.assign(static_cast<std::size_t>(w.hidden) * w.intermediate, 0.01F);
    layer.fc2_bias.assign(w.hidden, 0.0F);
    w.layers.push_back(std::move(layer));
    w.merger_norm_weight.assign(w.hidden, 1.0F);
    w.merger_norm_bias.assign(w.hidden, 0.0F);
    w.merger_fc1.assign(static_cast<std::size_t>(w.merger_width()) * w.merger_width(), 0.01F);
    w.merger_fc1_bias.assign(w.merger_width(), 0.0F);
    w.merger_fc2.assign(static_cast<std::size_t>(w.output_hidden) * w.merger_width(), 0.01F);
    w.merger_fc2_bias.assign(w.output_hidden, 0.0F);

    VisionItemControl ctrl;
    ctrl.patch_count            = 2;
    ctrl.merged_count           = 2;
    ctrl.segment_length         = 2;
    ctrl.position_ids           = {0, 0, 0, 1};
    ctrl.position_table_indices = {0, 0, 0, 0, 0, 0, 0, 0};
    ctrl.position_table_weights = {1, 0, 0, 0, 1, 0, 0, 0};

    std::vector<std::uint16_t> patches(static_cast<std::size_t>(ctrl.patch_count) * w.patch_width,
                                       bf16(0.5F));
    patches[3] = 0x7F81; // NaN patch input
    bool threw = false;
    try {
        (void)ninfer::models::qwen3_5::execution::encode_vision_on_cpu(w, patches, ctrl, 1);
    } catch (const std::runtime_error&) { threw = true; }
    expect(threw, "encode_vision_on_cpu must reject nonfinite output");
}

void test_degenerate_sizes(const cv::Kernels& kernels) {
    constexpr float kPoison = -321.5F;
    std::vector<float> buf(256, kPoison);
    std::vector<float> dummy(256, 1.0F);
    kernels.pack_panel(dummy.data(), 1, 0, 0, 0, buf.data());
    kernels.gemm_packed_bias(dummy.data(), 1, dummy.data(), 0, 4, dummy.data(), 4, buf.data(), 4);
    kernels.gemm_packed_bias(dummy.data(), 1, dummy.data(), 4, 4, dummy.data(), 0, buf.data(), 4);
    kernels.layer_norm_cols(dummy.data(), 4, 72, dummy.data(), dummy.data(), 0, 0, 1e-6F, buf.data());
    kernels.add_into(buf.data(), dummy.data(), 0);
    kernels.gelu_tanh(buf.data(), 0);
    kernels.rope_pair(buf.data(), buf.data() + 64, dummy.data(), dummy.data(), 0);
    kernels.segment_attention_block(dummy.data(), 16, 72, 0, 0, 0, 0, 8, 1.0F, dummy.data(),
                                    buf.data(), 16);
    kernels.segment_attention_block(dummy.data(), 16, 72, 0, 0, 0, -1, 8, 1.0F, dummy.data(),
                                    buf.data(), 16);
    kernels.segment_attention_block(dummy.data(), 16, 72, 0, 0, 0, 1, 0, 1.0F, dummy.data(),
                                    buf.data(), 16);
    const bool untouched = std::all_of(buf.begin(), buf.end(), [&](float v) { return v == kPoison; });
    expect(untouched, std::string(kernels.name) + " degenerate size wrote to output");

    // k == 0 must fill every token with bias[row], matching origin/main linear_tile behavior.
    const std::array<float, 3> bias0{1.25F, -2.5F, 0.75F};
    std::vector<float> y0(3 * 5, kPoison);
    kernels.gemm_packed_bias(dummy.data(), 1, bias0.data(), 3, 0, dummy.data(), 5, y0.data(), 5);
    for (int r = 0; r < 3; ++r) {
        for (int t = 0; t < 5; ++t) {
            expect(y0[static_cast<std::size_t>(r) * 5 + t] == bias0[static_cast<std::size_t>(r)],
                   std::string(kernels.name) + " gemm_packed_bias k=0 did not write bias");
        }
    }
}

#if defined(NINFER_VISION_CPU_X86)
void test_avx1_equivalence(std::mt19937& rng) {
    const cv::Kernels& port = cv::portable_kernels();
    const cv::Kernels& avx  = cv::avx_kernels();
    std::normal_distribution<float> normal(0.0F, 1.0F);

    // Exact bitwise equality for pack_panel, gemm_packed_bias, add_into, rope_pair.
    const int n = 13, k = 19, tokens = 25;
    std::vector<float> w(static_cast<std::size_t>(n) * k), x(static_cast<std::size_t>(k) * tokens),
        bias(n);
    for (float& v : w) { v = normal(rng); }
    for (float& v : x) { v = normal(rng); }
    for (float& v : bias) { v = normal(rng); }

    const int panels = (tokens + 15) / 16;
    std::vector<float> p_port(static_cast<std::size_t>(panels) * k * 16, 0.0F);
    std::vector<float> p_avx(static_cast<std::size_t>(panels) * k * 16, 0.0F);
    for (int p = 0; p < panels; ++p) {
        const int tok = p * 16, width = std::min(16, tokens - tok);
        port.pack_panel(x.data(), tokens, k, tok, width, p_port.data() + static_cast<std::size_t>(p) * k * 16);
        avx.pack_panel(x.data(), tokens, k, tok, width, p_avx.data() + static_cast<std::size_t>(p) * k * 16);
    }
    expect(p_port == p_avx, "pack_panel portable vs avx bitwise mismatch");

    std::vector<float> y_port(static_cast<std::size_t>(n) * tokens),
        y_avx(static_cast<std::size_t>(n) * tokens);
    port.gemm_packed_bias(w.data(), k, bias.data(), n, k, p_port.data(), tokens, y_port.data(), tokens);
    avx.gemm_packed_bias(w.data(), k, bias.data(), n, k, p_avx.data(), tokens, y_avx.data(), tokens);
    expect(std::memcmp(y_port.data(), y_avx.data(), y_port.size() * sizeof(float)) == 0,
           "gemm_packed_bias portable vs avx bitwise mismatch");

    auto a_port = y_port, a_avx = y_avx;
    port.add_into(a_port.data(), y_port.data(), a_port.size());
    avx.add_into(a_avx.data(), y_avx.data(), a_avx.size());
    expect(std::memcmp(a_port.data(), a_avx.data(), a_port.size() * sizeof(float)) == 0,
           "add_into portable vs avx bitwise mismatch");

    std::vector<float> rf_p(tokens), rs_p(tokens), cos_v(tokens), sin_v(tokens);
    for (int t = 0; t < tokens; ++t) {
        rf_p[t]  = normal(rng);
        rs_p[t]  = normal(rng);
        cos_v[t] = std::cos(0.1F * t);
        sin_v[t] = std::sin(0.1F * t);
    }
    auto rf_a = rf_p, rs_a = rs_p;
    port.rope_pair(rf_p.data(), rs_p.data(), cos_v.data(), sin_v.data(), tokens);
    avx.rope_pair(rf_a.data(), rs_a.data(), cos_v.data(), sin_v.data(), tokens);
    expect(std::memcmp(rf_p.data(), rf_a.data(), rf_p.size() * sizeof(float)) == 0 &&
               std::memcmp(rs_p.data(), rs_a.data(), rs_p.size() * sizeof(float)) == 0,
           "rope_pair portable vs avx bitwise mismatch");
}
#endif

void run_suite(const cv::Kernels& kernels) {
    std::mt19937 rng(20261119);
    test_gemm_packed_bias(kernels, rng);
    test_layer_norm_cols(kernels, rng);
    test_rope_and_gelu(kernels, rng);
    test_segment_attention(kernels, rng);
    test_nonfinite_preservation(kernels);
    test_degenerate_sizes(kernels);
}

} // namespace

int main() {
    run_suite(cv::portable_kernels());
    std::cout << "selected kernels: " << cv::kernels().name << '\n';
#if defined(NINFER_VISION_CPU_X86)
    if (&cv::kernels() == &cv::avx_kernels() || &cv::kernels() == &cv::avx2_kernels()) {
        std::mt19937 rng(20261119);
        run_suite(cv::avx_kernels());
        test_avx1_equivalence(rng);
    }
    if (&cv::kernels() == &cv::avx2_kernels()) { run_suite(cv::avx2_kernels()); }
#endif
    test_encode_isfinite_gate();
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " qwen3.5 vision CPU kernels\n";
    return failures == 0 ? 0 : 1;
}
