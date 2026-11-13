// Evaluation microbenchmark comparing Scalar/Portable vs AVX 1.0 paths for:
// 1. Native MoE CPU experts (moe_expert_cpu)
// 2. Gemma 4 CPU vision kernels (cpu_vision)

#include "models/gemma4/vision_cpu_kernels.h"
#include "ninfer/ops/moe_expert_cpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
namespace cv = ninfer::models::gemma4::cpu_vision;

constexpr int H = 2560, I = 640;

std::uint16_t float_to_bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fffu + ((word >> 16) & 1u);
    return static_cast<std::uint16_t>(word >> 16);
}

struct NativeBank {
    Weight weight;
    std::vector<std::uint8_t> data, high, scales;

    NativeBank(QType format, int rows, int columns, bool panel, std::mt19937& random) {
        weight.qtype = format;
        weight.n = rows;
        weight.k = columns;
        weight.scale_dtype = DType::FP16;
        if (format == QType::BF16) {
            weight.layout = QuantLayout::Contiguous;
            data.resize(std::size_t(rows) * columns * 2);
            for (std::size_t i = 0; i < std::size_t(rows) * columns; ++i) {
                const auto word = float_to_bf16((int(random() % 2001) - 1000) * 0.000025F);
                data[i * 2] = word & 255;
                data[i * 2 + 1] = word >> 8;
            }
            weight.qdata = data.data();
            return;
        }
        const int bits = format == QType::Q2_G64_FP16 ? 2 :
                         format == QType::Q4_G64_FP16 ? 4 :
                         format == QType::Q5_G64_FP16 ? 5 :
                         format == QType::Q6_G64_FP16 ? 6 : 8;
        const int group = bits == 8 ? 32 : 64;
        const int groups = columns / group;
        const int low_bytes = bits == 2 ? 16 : 32;
        const int high_bytes = bits == 5 ? 8 : bits == 6 ? 16 : 0;
        weight.group = group;
        weight.layout = panel ? QuantLayout::RowSplitPanel : QuantLayout::RowSplit;
        data.resize(std::size_t(rows) * groups * low_bytes);
        high.resize(std::size_t(rows) * groups * high_bytes);
        scales.resize(std::size_t(rows) * groups * 2);
        for (auto& b : data) { b = static_cast<std::uint8_t>(random()); }
        for (auto& b : high) { b = static_cast<std::uint8_t>(random()); }
        for (int row = 0; row < rows; ++row) {
            for (int g = 0; g < groups; ++g) {
                const unsigned scale = bits == 2 ? 0x2400U : bits == 8 ? 0x0800U : 0x1400U;
                const std::size_t s = (std::size_t(row) * groups + g) * 2;
                scales[s] = scale & 255;
                scales[s + 1] = scale >> 8;
            }
        }
        weight.qdata = data.data();
        weight.qhigh = high.empty() ? nullptr : high.data();
        weight.scales = scales.data();
    }
};

template <class Fn>
double measure_median_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { fn(); }
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void bench_native_moe(std::mt19937& rng, int iters) {
    const bool has_avx = cpu_expert_backend_available(CpuExpertBackend::Avx);
    std::cout << "=== Native MoE CPU Expert Benchmark (avx=" << has_avx << ") ===\n";
    struct Case {
        QType fmt;
        const char* name;
        bool a8;
    };
    const std::vector<Case> cases = {
        {QType::Q2_G64_FP16, "Q2_A8",   true},
        {QType::Q4_G64_FP16, "Q4_A8",   true},
        {QType::Q5_G64_FP16, "Q5_A8",   true},
        {QType::Q8_G32_FP16, "Q8_A8",   true},
        {QType::Q4_G64_FP16, "Q4_A16",  false},
        {QType::BF16,        "BF16",    false},
    };
    CpuExpertWorkspace ws(8);
    std::normal_distribution<float> normal(0.0F, 0.5F);
    for (const auto& c : cases) {
        NativeBank gate(c.fmt, I, H, true, rng);
        NativeBank up(c.fmt, I, H, true, rng);
        NativeBank down(c.fmt, H, I, true, rng);
        const CpuExpertWeights weights{{gate.weight, c.a8}, {up.weight, c.a8}, {down.weight, c.a8}};
        for (int T : {1, 8}) {
            std::vector<std::uint16_t> x(std::size_t(T) * H);
            for (auto& v : x) { v = float_to_bf16(normal(rng)); }
            std::vector<float> y(std::size_t(T) * H);
            const double ms_scalar = measure_median_ms(
                [&] { moe_expert_cpu(x, weights, ws, y, CpuExpertBackend::Scalar); }, 2, iters);
            double ms_avx = 0.0;
            if (has_avx) {
                ms_avx = measure_median_ms(
                    [&] { moe_expert_cpu(x, weights, ws, y, CpuExpertBackend::Avx); }, 2, iters);
            }
            std::cout << std::left << std::setw(8) << c.name << " T=" << T
                      << " | scalar=" << std::fixed << std::setprecision(4) << ms_scalar << " ms";
            if (has_avx) {
                std::cout << " avx=" << ms_avx << " ms (" << std::setprecision(2)
                          << (ms_scalar / ms_avx) << "x)";
            }
            std::cout << '\n';
        }
    }
}

void bench_vision(std::mt19937& rng, int iters) {
    const auto& portable = cv::portable_kernels();
#if defined(NINFER_VISION_CPU_X86)
    const auto& avx = cv::avx_kernels();
#endif
    std::cout << "=== Gemma 4 CPU Vision Kernels Benchmark ===\n";
    std::normal_distribution<float> normal(0.0F, 1.0F);

    struct GemmShape {
        int n, k, tokens;
    };
    for (const GemmShape s : {GemmShape{72, 2340, 128}, GemmShape{1152, 1152, 128}, GemmShape{4608, 1152, 64}}) {
        std::vector<float> w(std::size_t(s.n) * s.k), x(std::size_t(s.k) * s.tokens),
            y(std::size_t(s.n) * s.tokens);
        for (float& v : w) { v = normal(rng); }
        for (float& v : x) { v = normal(rng); }
        const double ms_port = measure_median_ms(
            [&] { portable.gemm(w.data(), s.k, s.n, s.k, x.data(), s.tokens, s.tokens, y.data(), s.tokens); },
            2, iters);
        std::cout << "gemm " << s.n << "x" << s.k << "x" << s.tokens
                  << " | portable=" << std::fixed << std::setprecision(4) << ms_port << " ms";
#if defined(NINFER_VISION_CPU_X86)
        const double ms_avx = measure_median_ms(
            [&] { avx.gemm(w.data(), s.k, s.n, s.k, x.data(), s.tokens, s.tokens, y.data(), s.tokens); },
            2, iters);
        std::cout << " avx=" << ms_avx << " ms (" << std::setprecision(2) << (ms_port / ms_avx) << "x)";
#endif
        std::cout << '\n';
    }

    // Softmax columns
    {
        const int rows = 256, cols = 256;
        std::vector<float> s(rows * cols), sums(cols);
        for (float& v : s) { v = normal(rng) * 5.0F; }
        const double ms_port = measure_median_ms(
            [&] { portable.softmax_columns(s.data(), cols, rows, cols, sums.data()); }, 3, iters * 5);
        std::cout << "softmax 256x256 | portable=" << std::setprecision(4) << ms_port << " ms";
#if defined(NINFER_VISION_CPU_X86)
        const double ms_avx = measure_median_ms(
            [&] { avx.softmax_columns(s.data(), cols, rows, cols, sums.data()); }, 3, iters * 5);
        std::cout << " avx=" << ms_avx << " ms (" << std::setprecision(2) << (ms_port / ms_avx) << "x)";
#endif
        std::cout << '\n';
    }

    // GeGLU
    {
        const std::size_t count = 4608 * 64;
        std::vector<float> g(count), u(count), out(count);
        for (std::size_t i = 0; i < count; ++i) {
            g[i] = normal(rng);
            u[i] = normal(rng);
        }
        const double ms_port = measure_median_ms(
            [&] { portable.geglu(g.data(), u.data(), out.data(), count); }, 3, iters * 5);
        std::cout << "geglu n=" << count << " | portable=" << std::setprecision(4) << ms_port << " ms";
#if defined(NINFER_VISION_CPU_X86)
        const double ms_avx = measure_median_ms(
            [&] { avx.geglu(g.data(), u.data(), out.data(), count); }, 3, iters * 5);
        std::cout << " avx=" << ms_avx << " ms (" << std::setprecision(2) << (ms_port / ms_avx) << "x)";
#endif
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 15;
    if (argc > 1) { iters = std::max(1, std::stoi(argv[1])); }
    std::mt19937 rng(20261113);
    bench_native_moe(rng, iters);
    bench_vision(rng, iters);
    return 0;
}
