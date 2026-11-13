// Microbenchmark for host GGUF MoE CPU experts (gguf_expert_cpu and gguf_expert_cpu_ggml)
// across Flash-Next projection shapes, token counts (T=1,4,8), and worker thread counts.

#include "core/weight.h"
#include "ops/moe_experts/gguf_expert_cpu.h"
#include "ops/moe_experts/gguf_expert_cpu_ggml.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;

std::uint16_t float_to_bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fffu + ((word >> 16) & 1u);
    return static_cast<std::uint16_t>(word >> 16);
}

std::uint16_t half_bits(float value) {
    const auto word = std::bit_cast<std::uint32_t>(value);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    const std::uint32_t mantissa = ((word & 0x7fffffu) + 0x1000u) >> 13;
    return static_cast<std::uint16_t>((exponent << 10) + mantissa);
}

struct BenchMatrix {
    QType format;
    const char* name;
    int rows;
    int k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
};

BenchMatrix make_matrix(QType format, const char* name, int rows, int k, std::mt19937& rng) {
    int group = 0;
    int block = 0;
    if (format == QType::GGUF_Q2_0) {
        group = 64;
        block = 18;
    } else if (format == QType::GGUF_Q8_0) {
        group = 32;
        block = 34;
    } else {
        const auto shape = gguf_block_shape(format);
        group = shape.elements;
        block = shape.bytes;
    }
    BenchMatrix m{format, name, rows, k, std::int64_t(k / group) * block, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    for (auto& b : m.bytes) { b = static_cast<std::uint8_t>(rng()); }
    std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d = half_bits(scale(rng));
            std::memcpy(p, &d, 2);
        }
    }
    return m;
}

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

void run_parallel_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, float* y, int threads,
                       GgufCpuBackend backend) {
    if (threads <= 1) {
        gguf_cpu_rows(w, x, 0, w.rows, y, std::size_t(w.rows), backend);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    const int chunk = (w.rows + threads - 1) / threads;
    for (int t = 1; t < threads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(w.rows, begin + chunk);
        if (begin < end) {
            workers.emplace_back([&, begin, end] {
                gguf_cpu_rows(w, x, begin, end, y, std::size_t(w.rows), backend);
            });
        }
    }
    gguf_cpu_rows(w, x, 0, std::min(w.rows, chunk), y, std::size_t(w.rows), backend);
    for (auto& worker : workers) { worker.join(); }
}

void run_parallel_ggml(const GgufCpuMatrix& w, const GgmlCpuActivation& x, float* y, int threads) {
    if (threads <= 1) {
        gguf_cpu_rows_ggml(w, x, 0, w.rows, y, std::size_t(w.rows));
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    const int chunk = (w.rows + threads - 1) / threads;
    for (int t = 1; t < threads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(w.rows, begin + chunk);
        if (begin < end) {
            workers.emplace_back([&, begin, end] {
                gguf_cpu_rows_ggml(w, x, begin, end, y, std::size_t(w.rows));
            });
        }
    }
    gguf_cpu_rows_ggml(w, x, 0, std::min(w.rows, chunk), y, std::size_t(w.rows));
    for (auto& worker : workers) { worker.join(); }
}

}  // namespace

int main(int argc, char** argv) {
    int warmup = 3;
    int iters = 15;
    if (argc > 1) { iters = std::max(1, std::stoi(argv[1])); }

    std::mt19937 rng(20261113);
    std::normal_distribution<float> normal(0.0F, 1.0F);

    struct Spec {
        QType format;
        const char* name;
        int rows;
        int k;
    };
    const std::vector<Spec> specs = {
        {QType::GGUF_IQ3_S,  "IQ3_S",  640,  2560},
        {QType::GGUF_IQ4_NL, "IQ4_NL", 640,  2560},
        {QType::GGUF_Q8_0,   "Q8_0",   640,  2560},
        {QType::GGUF_IQ2_S,  "IQ2_S",  640,  2560},
        {QType::GGUF_Q2_0,   "Q2_0",   640,  2560},
        {QType::GGUF_IQ4_NL, "IQ4_NL", 2560, 640},
        {QType::GGUF_Q8_0,   "Q8_0",   2560, 640},
        {QType::GGUF_Q2_0,   "Q2_0",   2560, 640},
    };

    const bool has_avx = gguf_cpu_backend_available(GgufCpuBackend::Avx);
    const bool has_avx2 = gguf_cpu_backend_available(GgufCpuBackend::Avx2);
    const bool has_ggml = gguf_ggml_cpu_available();

    std::cout << "GGUF CPU Expert Benchmark (avx=" << has_avx
              << ", avx2=" << has_avx2 << ", ggml=" << has_ggml
              << ", iters=" << iters << ")\n";
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& s : specs) {
        const BenchMatrix m = make_matrix(s.format, s.name, s.rows, s.k, rng);
        const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};

        for (int tokens : {1, 4, 8}) {
            std::vector<std::uint16_t> x_bf16(std::size_t(tokens) * s.k);
            for (auto& v : x_bf16) { v = float_to_bf16(normal(rng)); }

            GgufCpuActivation act(tokens, s.k);
            act.prepare_bf16(s.format, x_bf16.data(), s.k, tokens);

            std::vector<float> y(std::size_t(tokens) * s.rows, 0.0F);

            for (int threads : {1, 6, 8}) {
                const double ms_scalar = measure_median_ms(
                    [&] { run_parallel_rows(w, act, y.data(), threads, GgufCpuBackend::Scalar); },
                    warmup, iters);
                double ms_avx = 0.0;
                if (has_avx) {
                    ms_avx = measure_median_ms(
                        [&] { run_parallel_rows(w, act, y.data(), threads, GgufCpuBackend::Avx); },
                        warmup, iters);
                }
                double ms_ggml = 0.0;
                if (has_ggml && gguf_ggml_supported(s.format)) {
                    const GgmlActLayout layout =
                        (s.format == QType::GGUF_IQ4_NL || s.format == QType::GGUF_Q8_0)
                            ? GgmlActLayout::Q8_0
                            : GgmlActLayout::Q8K;
                    if (layout == GgmlActLayout::Q8_0 || s.k % 256 == 0) {
                        GgmlCpuActivation gact(tokens, s.k);
                        gact.prepare_bf16(layout, x_bf16.data(), s.k, tokens);
                        ms_ggml = measure_median_ms(
                            [&] { run_parallel_ggml(w, gact, y.data(), threads); }, warmup, iters);
                    }
                }
                std::cout << std::left << std::setw(7) << s.name
                          << std::setw(10) << (std::to_string(s.rows) + "x" + std::to_string(s.k))
                          << " T=" << tokens << " thr=" << threads
                          << " | scalar=" << ms_scalar << " ms";
                if (has_avx) {
                    std::cout << " avx=" << ms_avx << " ms (" << std::setprecision(2)
                              << (ms_scalar / ms_avx) << "x)" << std::setprecision(4);
                }
                if (ms_ggml > 0.0) {
                    std::cout << " ggml=" << ms_ggml << " ms (" << std::setprecision(2)
                              << (ms_scalar / ms_ggml) << "x)" << std::setprecision(4);
                }
                std::cout << '\n';
            }
        }
    }

    // SwiGLU BF16 microbenchmark across expert intermediate count (640 * 8 = 5120)
    const std::size_t n_swiglu = 640 * 8;
    std::vector<float> gate(n_swiglu), up(n_swiglu);
    std::vector<std::uint16_t> out(n_swiglu);
    for (std::size_t i = 0; i < n_swiglu; ++i) {
        gate[i] = normal(rng);
        up[i] = normal(rng);
    }
    const double ms_sw_scalar = measure_median_ms(
        [&] { gguf_cpu_swiglu_bf16(gate.data(), up.data(), n_swiglu, out.data(), GgufCpuBackend::Scalar); },
        10, iters * 10);
    std::cout << "SwiGLU n=" << n_swiglu << " | scalar=" << ms_sw_scalar << " ms";
    if (has_avx) {
        const double ms_sw_avx = measure_median_ms(
            [&] { gguf_cpu_swiglu_bf16(gate.data(), up.data(), n_swiglu, out.data(), GgufCpuBackend::Avx); },
            10, iters * 10);
        std::cout << " avx=" << ms_sw_avx << " ms (" << std::setprecision(2)
                  << (ms_sw_scalar / ms_sw_avx) << "x)";
    }
    std::cout << '\n';
    return 0;
}
