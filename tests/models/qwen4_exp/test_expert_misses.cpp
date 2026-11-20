#include "models/qwen4_exp/cpu_node_coordinator.h"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::models::qwen4_exp;

void require(bool ok, const char* msg) {
    if (!ok) { throw std::runtime_error(msg); }
}

// T-10: Primary throwing an exception while remote socket SpinPools are still running MUST drain
// every launched remote runner (done_seq == expected_seq) before returning, so reusing/mutating
// shared activation buffers on subsequent calls never races under ThreadSanitizer.
void test_exception_drains_remote_pools() {
    SpinPool remote_pool(2);
    SpinPool primary_pool(2);
    remote_pool.spin(true);
    primary_pool.spin(true);

    struct SharedWork {
        std::vector<float> buf;
        SpinPool* pool = nullptr;
        std::atomic<bool> started{false};
        double sum = 0.0;
    } work;
    work.buf.assign(512, 1.0F);
    work.pool = &remote_pool;

    RemoteNodeRunner remote;
    remote.thread = std::thread([&] {
        remote.loop(-1, -1, [](const void* payload) {
            auto* w = const_cast<SharedWork*>(static_cast<const SharedWork*>(payload));
            w->started.store(true, std::memory_order_release);
            double local_sum = 0.0;
            std::atomic<int> acc_guard{0};
            w->pool->run(64, [&](std::size_t i) {
                std::this_thread::sleep_for(std::chrono::microseconds(250));
                float s = 0.0F;
                for (float v : w->buf) { s += v; }
                (void)s;
                acc_guard.fetch_add(int(i == 0), std::memory_order_relaxed);
            });
            w->sum = local_sum;
        });
    });

    struct StopGuard {
        RemoteNodeRunner& r;
        ~StopGuard() {
            {
                std::lock_guard lock(r.mutex);
                r.stop.store(true, std::memory_order_release);
            }
            r.cv.notify_all();
            if (r.thread.joinable()) { r.thread.join(); }
        }
    } guard{remote};

    const std::array<RemoteNodeRunner*, 1> remotes{&remote};
    std::atomic<bool> never_cancel{false};

    for (int round = 0; round < 3; ++round) {
        work.started.store(false, std::memory_order_release);
        std::fill(work.buf.begin(), work.buf.end(), float(round + 1));

        bool caught_primary = false;
        std::string message;
        try {
            dispatch_cpu_nodes(
                /*primary_active=*/true, remotes, &work,
                [&] {
                    while (!work.started.load(std::memory_order_acquire)) { spin_hint(); }
                    primary_pool.run(4, [&](std::size_t idx) {
                        if (idx == 0) { throw std::runtime_error("primary injected fault"); }
                    });
                },
                [&] { return never_cancel.load(std::memory_order_acquire); });
        } catch (const std::runtime_error& err) {
            caught_primary = true;
            message        = err.what();
        }
        require(caught_primary && message == "primary injected fault",
                "T-10 expected primary exception to propagate");
        require(remote.done_seq.load(std::memory_order_acquire) == remote.expected_seq,
                "T-10 remote runner must finish (done_seq == expected_seq) before dispatch returns");

        // Immediately overwrite shared buffer (simulating next call's prepare_bf16 / resize):
        // under TSan this catches any lingering unsynchronized remote worker.
        std::fill(work.buf.begin(), work.buf.end(), -999.0F);
        work.buf.resize(256);
        work.buf.resize(512, 2.0F);
    }
}

// T-20: Cancellation while remote SpinPool is running completes in bounded time and the
// chronologically first exception across nodes is never overwritten by a later error/cancel.
void test_cancel_while_remote_running_and_first_error() {
    // Part A: Chronologically first remote error is preserved even when another node or cancel
    // fails afterwards.
    {
        std::atomic<bool> r1_failed{false};
        RemoteNodeRunner r1, r2;
        r1.thread = std::thread([&] {
            r1.loop(-1, -1, [&](const void*) {
                r1_failed.store(true, std::memory_order_release);
                throw std::runtime_error("remote1 disk error");
            });
        });
        r2.thread = std::thread([&] {
            r2.loop(-1, -1, [&](const void*) {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                throw std::runtime_error("remote2 secondary error");
            });
        });
        struct Guard2 {
            RemoteNodeRunner &a, &b;
            ~Guard2() {
                for (auto* r : {&a, &b}) {
                    {
                        std::lock_guard lock(r->mutex);
                        r->stop.store(true, std::memory_order_release);
                    }
                    r->cv.notify_all();
                    if (r->thread.joinable()) { r->thread.join(); }
                }
            }
        } g{r1, r2};

        const std::array<RemoteNodeRunner*, 2> remotes{&r1, &r2};
        std::string what;
        try {
            dispatch_cpu_nodes(
                true, remotes, nullptr,
                [&] {
                    while (!r1_failed.load(std::memory_order_acquire)) { spin_hint(); }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    throw std::runtime_error("primary late error");
                },
                [] { return false; });
        } catch (const std::runtime_error& e) { what = e.what(); }
        require(what == "remote1 disk error",
                "T-20 chronologically first remote error was overwritten");
    }

    // Part B: Cancellation aborts a long chunked remote SpinPool batch in bounded time.
    {
        SpinPool pool(2);
        pool.spin(true);
        std::atomic<bool> cancelled{false};
        std::atomic<int> executed{0};
        std::atomic<bool> entered{false};

        RemoteNodeRunner remote;
        remote.thread = std::thread([&] {
            remote.loop(-1, -1, [&](const void*) {
                // 2000 items * 2ms = ~4 seconds if cancellation were ignored.
                pool.run(
                    2000,
                    [&](std::size_t) {
                        entered.store(true, std::memory_order_release);
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                        executed.fetch_add(1, std::memory_order_relaxed);
                    },
                    &cancelled);
            });
        });
        struct Guard1 {
            RemoteNodeRunner& r;
            ~Guard1() {
                {
                    std::lock_guard lock(r.mutex);
                    r.stop.store(true, std::memory_order_release);
                }
                r.cv.notify_all();
                if (r.thread.joinable()) { r.thread.join(); }
            }
        } g{remote};

        std::thread canceller([&] {
            while (!entered.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            cancelled.store(true, std::memory_order_release);
        });

        const auto t0 = std::chrono::steady_clock::now();
        bool saw_cancel = false;
        const std::array<RemoteNodeRunner*, 1> remotes{&remote};
        try {
            dispatch_cpu_nodes(
                false, remotes, nullptr, [] {},
                [&] { return cancelled.load(std::memory_order_acquire); });
        } catch (const ninfer::ops::CpuExpertCancelled&) {
            saw_cancel = true;
        }
        canceller.join();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        require(saw_cancel, "T-20 cancellation must throw CpuExpertCancelled");
        require(elapsed.count() < 1000,
                "T-20 cancellation took too long (remote SpinPool did not abort)");
        require(executed.load() < 500, "T-20 remote SpinPool ran nearly all tasks despite cancel");
        require(remote.done_seq.load(std::memory_order_acquire) == remote.expected_seq,
                "T-20 remote done_seq did not reach expected_seq after cancel");
    }
}

} // namespace

int main() {
    try {
        test_exception_drains_remote_pools();
        test_cancel_while_remote_running_and_first_error();
        std::cout << "qwen4_exp_expert_misses: OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
