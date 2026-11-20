#pragma once

#include "core/numa.h"
#include "ninfer/ops/moe_expert_cpu.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace ninfer::models::qwen4_exp {

// One turn of a spin wait: the processor's pause hint where it has one, a yield elsewhere.
// YieldProcessor() is the Windows SDK's portable spelling; the GCC/Clang builtin is not known to
// nvcc's front end in MSVC mode.
inline void spin_hint() {
#if defined(_WIN32)
    YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

struct FirstFailure {
    std::mutex mutex;
    std::exception_ptr error;

    void record(std::exception_ptr e) noexcept {
        if (!e) { return; }
        std::lock_guard lock(mutex);
        if (!error) { error = std::move(e); }
    }
};

// Workers that spin while a batch is likely (between spin(true) and spin(false)) and sleep
// otherwise. run() executes job(i) for every i < count on the workers and the calling thread and
// returns when all are done, rethrowing the first failure. A claim is one CAS on a ticket holding
// the run's generation and next index, so a worker still leaving an earlier run can never claim an
// index of a later one.
class SpinPool {
public:
    explicit SpinPool(std::size_t threads, std::vector<int> pins = {}) {
        for (std::size_t i = 0; i < threads; ++i) {
            const int pin = i < pins.size() ? pins[i] : -1;
            workers_.emplace_back([this, pin] { work(pin); });
        }
    }
    ~SpinPool() {
        {
            std::lock_guard lock(mutex_);
            stop_.store(true);
        }
        wake_.notify_all();
        for (auto& worker : workers_) { worker.join(); }
    }
    SpinPool(const SpinPool&)            = delete;
    SpinPool& operator=(const SpinPool&) = delete;

    void spin(bool on) {
        {
            std::lock_guard lock(mutex_);
            spinning_.store(on);
        }
        wake_.notify_all();
    }

    template <class Job>
    void run(std::size_t count, Job&& job,
             const std::atomic<bool>* cancelled_a = nullptr,
             const std::atomic<bool>* cancelled_b = nullptr) {
        if (count == 0) { return; }
        context_     = &job;
        call_        = [](void* context, std::size_t i) {
            (*static_cast<std::remove_reference_t<Job>*>(context))(i);
        };
        cancelled_a_ = cancelled_a;
        cancelled_b_ = cancelled_b;
        const std::uint64_t generation = (ticket_.load(std::memory_order_relaxed) >> 32) + 1;
        done_.store(0, std::memory_order_relaxed);
        // A worker claims only when the ticket's generation matches this one.
        size_.store(generation << 32 | count, std::memory_order_release);
        {
            std::lock_guard lock(mutex_);
            ticket_.store(generation << 32, std::memory_order_release);
        }
        wake_.notify_all();
        drain();
        while (done_.load(std::memory_order_acquire) != count) { spin_hint(); }
        cancelled_a_ = nullptr;
        cancelled_b_ = nullptr;
        if (failure_) {
            auto failure = std::exchange(failure_, nullptr);
            std::rethrow_exception(failure);
        }
    }

private:
    bool is_cancelled() const noexcept {
        return (cancelled_a_ && cancelled_a_->load(std::memory_order_acquire)) ||
               (cancelled_b_ && cancelled_b_->load(std::memory_order_acquire));
    }

    void drain() {
        std::uint64_t ticket = ticket_.load(std::memory_order_acquire);
        for (;;) {
            const std::uint64_t size = size_.load(std::memory_order_acquire);
            const std::size_t i      = std::size_t(ticket & 0xffffffffULL);
            if ((size >> 32) != (ticket >> 32) || i >= (size & 0xffffffffULL)) { return; }
            if (!ticket_.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
                continue;
            }
            if (is_cancelled()) {
                std::lock_guard lock(failure_mutex_);
                if (!failure_) { failure_ = std::make_exception_ptr(ops::CpuExpertCancelled()); }
            } else {
                try {
                    call_(context_, i);
                } catch (...) {
                    std::lock_guard lock(failure_mutex_);
                    if (!failure_) { failure_ = std::current_exception(); }
                }
            }
            done_.fetch_add(1, std::memory_order_release);
            ticket = ticket_.load(std::memory_order_acquire);
        }
    }

    void work(int pin_cpu) {
        if (pin_cpu >= 0) { (void)bind_thread_to_cpu(pin_cpu); }
        std::uint64_t seen = 0;
        for (;;) {
            if (stop_.load()) { return; }
            const std::uint64_t generation = ticket_.load(std::memory_order_acquire) >> 32;
            if (generation != seen) {
                seen = generation;
                drain();
                continue;
            }
            if (spinning_.load(std::memory_order_relaxed)) {
                spin_hint();
                continue;
            }
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] {
                return stop_.load() || spinning_.load() || (ticket_.load() >> 32) != seen;
            });
        }
    }

    std::vector<std::thread> workers_;
    void (*call_)(void*, std::size_t) = nullptr;
    void* context_                    = nullptr;
    const std::atomic<bool>* cancelled_a_ = nullptr;
    const std::atomic<bool>* cancelled_b_ = nullptr;
    std::atomic<std::uint64_t> ticket_{0}; // generation << 32 | next index
    std::atomic<std::uint64_t> size_{0};   // generation << 32 | count
    std::atomic<std::size_t> done_{0};
    std::atomic<bool> spinning_{false}, stop_{false};
    std::mutex mutex_, failure_mutex_;
    std::condition_variable wake_;
    std::exception_ptr failure_;
};

// Persistent per-socket remote supervisor thread synchronized via monotonic request/done sequence
// counters and a shared FirstFailure sink.
struct RemoteNodeRunner {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> spinning{false};
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> pending_seq{0};
    std::atomic<std::uint64_t> done_seq{0};
    std::uint64_t expected_seq = 0;
    const void* payload        = nullptr;
    std::atomic<FirstFailure*> error_sink{nullptr};
    std::exception_ptr failure;

    template <class Fn>
    void loop(int lead_cpu, int node_id, Fn&& fn) {
        if (lead_cpu >= 0) {
            (void)bind_thread_to_cpu(lead_cpu);
        } else if (node_id >= 0) {
            (void)bind_thread_to_node(node_id);
        }
        std::uint64_t seen = 0;
        for (;;) {
            if (stop.load(std::memory_order_acquire)) { return; }
            const std::uint64_t seq = pending_seq.load(std::memory_order_acquire);
            if (seq != seen) {
                seen = seq;
                try {
                    failure = nullptr;
                    fn(payload);
                } catch (...) {
                    std::exception_ptr ep = std::current_exception();
                    failure               = ep;
                    if (FirstFailure* sink = error_sink.load(std::memory_order_acquire)) {
                        sink->record(std::move(ep));
                    }
                }
                done_seq.store(seq, std::memory_order_release);
                continue;
            }
            if (spinning.load(std::memory_order_relaxed)) {
                spin_hint();
                continue;
            }
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] {
                return stop.load(std::memory_order_acquire) ||
                       spinning.load(std::memory_order_acquire) ||
                       pending_seq.load(std::memory_order_acquire) != seen;
            });
        }
    }
};

// Launches every active remote node, runs the primary node's callable inside try/catch, and
// unconditionally waits for every launched remote node to complete (done_seq == expected_seq)
// before rethrowing the chronologically first failure.
template <class RemoteRange, class PrimaryFn, class CancelFn>
void dispatch_cpu_nodes(bool primary_active, RemoteRange&& active_remotes, const void* payload,
                        PrimaryFn&& primary_fn, CancelFn&& is_cancelled) {
    FirstFailure first;
    for (RemoteNodeRunner* r : active_remotes) {
        r->failure = nullptr;
        r->payload = payload;
        r->error_sink.store(&first, std::memory_order_release);
        {
            std::lock_guard lock(r->mutex);
            ++r->expected_seq;
            r->pending_seq.store(r->expected_seq, std::memory_order_release);
        }
        r->cv.notify_one();
    }
    if (primary_active) {
        try {
            primary_fn();
        } catch (...) {
            first.record(std::current_exception());
        }
    }
    for (RemoteNodeRunner* r : active_remotes) {
        while (r->done_seq.load(std::memory_order_acquire) != r->expected_seq) {
            if (is_cancelled()) {
                std::this_thread::yield();
            } else {
                spin_hint();
            }
        }
        r->error_sink.store(nullptr, std::memory_order_release);
        if (r->failure) { first.record(std::exchange(r->failure, nullptr)); }
    }
    if (first.error) { std::rethrow_exception(first.error); }
}

} // namespace ninfer::models::qwen4_exp
